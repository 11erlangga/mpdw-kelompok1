import torch
import torch.nn as nn
import numpy as np

# Model Architecture for N-BEATS-I
class NBeatsBlock(nn.Module):
  def __init__(self,
                input_len,
                output_len,
                hidden_units_1,
                hidden_units_2,
                hidden_units_3,
                hidden_units_4,
                block_type,
                poly_degree=2,
                num_fourier=10,
                seasonality_period=252):
    
    super().__init__()
    self.input_len = input_len
    self.output_len = output_len
    self.block_type = block_type
    
    # 4-layer MLP
    self.fc = nn.Sequential(
        nn.Linear(input_len, hidden_units_1), 
        nn.ReLU(),
        
        nn.Linear(hidden_units_1, hidden_units_2), 
        nn.ReLU(),
        
        nn.Linear(hidden_units_2, hidden_units_3), 
        nn.ReLU(),
        
        nn.Linear(hidden_units_3, hidden_units_4), 
        nn.ReLU()
    )
    
    if block_type == 'trend':
      # Trend basis (polynomial)
      self.num_coeffs = poly_degree + 1
      
      # Basis: t_back = [0, ..., H-1], t_forecast = [0, ..., T-1]
      # Normalisasi waktu ke [0, 1] untuk stabilitas
      t_back = torch.linspace(-1, 0, input_len)
      t_fore = torch.linspace(0, 1, output_len)

      # Pangkat: [0, 1, ..., poly_degree]
      powers = torch.arange(self.num_coeffs).unsqueeze(1)
      
      # Basis: [p+1, H] dan [p+1, T]
      backcast_basis = (t_back.unsqueeze(0)) ** powers
      forecast_basis = (t_fore.unsqueeze(0)) ** powers
      
    elif block_type == 'seasonality':
      # Seasonality basis (Fourier)
      self.num_coeffs = 2 * num_fourier 
      
      # Waktu: t = [0, ..., H-1] dan [0, ..., T-1]
      t_back = torch.arange(input_len)
      t_fore = torch.arange(output_len)
      
      # Harmonics: n = [1, 2, ..., num_fourier]
      n = torch.arange(1, num_fourier + 1)
      
      # Frekuensi: 2 * pi * n / periode
      freq = 2 * np.pi * n / seasonality_period
      
      # Basis: [H, 2*N] dan [T, 2*N]
      backcast_basis = torch.cat([
        torch.cos(freq * t_back.unsqueeze(1)), 
        torch.sin(freq * t_back.unsqueeze(1))
        ], dim=1).T
      
      forecast_basis = torch.cat([
        torch.cos(freq * t_fore.unsqueeze(1)), 
        torch.sin(freq * t_fore.unsqueeze(1))
        ], dim=1).T
        
    else:
      raise ValueError(f"Tipe blok tidak dikenal: {block_type}")
        
    # Head untuk Koefisien
    # Layer ini memprediksi koefisien (theta)
    self.backcast_head = nn.Linear(hidden_units_4, self.num_coeffs)
    self.forecast_head = nn.Linear(hidden_units_4, self.num_coeffs)
    
    # Kaiming init untuk stabilitas
    for layer in self.fc:
      if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)

    nn.init.kaiming_uniform_(self.backcast_head.weight, nonlinearity='linear')
    nn.init.zeros_(self.backcast_head.bias)
    nn.init.kaiming_uniform_(self.forecast_head.weight, nonlinearity='linear')
    nn.init.zeros_(self.forecast_head.bias)
    
    # Daftarkan basis sebagai buffer (konstan, tidak butuh gradien)
    self.register_buffer('backcast_basis', backcast_basis)
    self.register_buffer('forecast_basis', forecast_basis)

  def forward(self, x):
    # x shape: [BATCH_SIZE, INPUT_LEN]
    
    # Input: [B, H] atau [B, H, 1]
    if x.dim() == 3:
      x = x.squeeze(-1)  # pastikan univariat

    hidden = self.fc(x)
    
    # Koefisien: [BATCH_SIZE, num_coeffs]
    theta_b = self.backcast_head(hidden)
    theta_f = self.forecast_head(hidden)
    
    # Buat backcast/forecast: [BATCH_SIZE, num_coeffs] @ [num_coeffs, H/T]
    # Hasil: [BATCH_SIZE, H/T]
    backcast = theta_b @ self.backcast_basis
    forecast = theta_f @ self.forecast_basis
    
    return backcast, forecast

class NBeatsStack(nn.Module):
  def __init__(self, num_blocks, input_len, output_len, hidden_units_1, hidden_units_2, hidden_units_3, hidden_units_4, block_type, **kwargs):
    super().__init__() 
    self.blocks = nn.ModuleList([
      NBeatsBlock(input_len, output_len, hidden_units_1, hidden_units_2, hidden_units_3, hidden_units_4, block_type, **kwargs) for _ in range(num_blocks)
      ])
    self.input_len = input_len
    self.output_len = output_len

  def forward(self, x):
    # x shape: [BATCH_SIZE, INPUT_LEN]
    residual = x
    
    # Inisialisasi total forecast dan backcast untuk stack ini
    total_backcast = torch.zeros(x.size(0), self.input_len, device=x.device)
    total_forecast = torch.zeros(x.size(0), self.output_len, device=x.device)

    for block in self.blocks:
      # Dapatkan backcast dan forecast dari blok saat ini
      backcast_i, forecast_i = block(residual)
      
      # Kurangi backcast dari residual (residual learning)
      residual = residual - backcast_i
      
      # Tambahkan forecast dan backcast ke total stack
      total_backcast += backcast_i
      total_forecast += forecast_i
        
    # Kembalikan total backcast (untuk stack berikutnya) dan total forecast (untuk prediksi akhir)
    return total_backcast, total_forecast, residual

class NBeatsInterpretableModel(nn.Module):
  def __init__(self,
               input_len,
               output_len,
               trend_blocks=3,
               trend_hiddens=[128, 128, 128, 128],
               poly_degree=3,
               season_blocks=3,
               season_hiddens=[128, 128, 128, 128],
               num_fourier=10,
               seasonality_period=252):
      
      super().__init__()
      
      # Stack 1: Trend
      self.trend_stack = NBeatsStack(
        num_blocks=trend_blocks,
        input_len=input_len,
        output_len=output_len,
        hidden_units_1=trend_hiddens[0],
        hidden_units_2=trend_hiddens[1],
        hidden_units_3=trend_hiddens[2],
        hidden_units_4=trend_hiddens[3],
        block_type='trend',
        poly_degree=poly_degree
      )
      
      # Stack 2: Seasonality
      self.seasonality_stack = NBeatsStack(
        num_blocks=season_blocks,
        input_len=input_len,
        output_len=output_len,
        hidden_units_1=season_hiddens[0],     
        hidden_units_2=season_hiddens[1],
        hidden_units_3=season_hiddens[2],
        hidden_units_4=season_hiddens[3],        
        block_type='seasonality',
        num_fourier=num_fourier,
        seasonality_period=seasonality_period
      )

  def forward(self, x):
   # Input [B, H, 1]
    x_in = x.squeeze(-1) # -> [B, H]
    
    # Stack Trend
    trend_backcast, trend_forecast, residual_1 = self.trend_stack(x_in)
    
    # Stack Seasonality
    # Prediksi komponen musiman dari residual
    seasonality_backcast, seasonality_forecast, residual_2 = self.seasonality_stack(residual_1)
    
    # Total Prediksi
    forecast_total = trend_forecast + seasonality_forecast

    return {
    "forecast_total": forecast_total,
    "forecast_trend": trend_forecast,
    "forecast_seasonality": seasonality_forecast,
    "residual": residual_2}
