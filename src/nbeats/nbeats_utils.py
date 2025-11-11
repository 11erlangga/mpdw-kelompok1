import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

# Custom Dataset untuk Time Series
class TimeSeriesDataset(Dataset):
  def __init__(self, data, input_len, output_len, transform=None):
      """
      data: np.array, [TotalTime, NumFeatures]
      input_len: int, panjang histori (X)
      output_len: int, panjang prediksi (y)
      """
      self.data = data
      self.input_len = input_len
      self.output_len = output_len
      
      # Hitung jumlah total 'window' yang valid yang bisa dibuat
      self.num_samples = len(data) - input_len - output_len + 1
      
      self.transform = transform

  def __len__(self):
    """
    Mengembalikan jumlah total 'window' yang bisa dibuat.
    """
    return self.num_samples

  def __getitem__(self, idx):
    """
    Mengambil satu 'window' (X, y) berdasarkan index.
    """
    # Tentukan titik awal dan akhir dari window
    start_idx_x = idx
    end_idx_x = idx + self.input_len
    
    start_idx_y = end_idx_x
    end_idx_y = start_idx_y + self.output_len
    
    # Ambil data X dan y
    X = self.data[start_idx_x:end_idx_x, :]
    y = self.data[start_idx_y:end_idx_y, :]
  
    y = y.squeeze(-1)
    
    # Ubah ke tensor
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    if self.transform:
      X_tensor, y_tensor = self.transform(X_tensor, y_tensor)
    return X_tensor, y_tensor
  
  def __repr__(self):
    return f"TimeSeriesDataset(n={len(self)}, input_len={self.input_len}, output_len={self.output_len})"

# Callback: EarlyStopping
class EarlyStopping:
  def __init__(self, patience=50, min_delta=1e-4, restore_best_model=True, path='best_model.pt'):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_loss = float('inf')
    self.early_stop = False
    
    self.restore_best_model = restore_best_model
    self.path = path

  def __call__(self, val_loss, model):
    if self.best_loss - val_loss > self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
      torch.save(model.cpu().state_dict(), self.path)
    else:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True
        if self.restore_best_model:
          model.load_state_dict(torch.load(self.path))

# Fungsi Plotting (Progression Plot)
def progression_plot(epoch_history, train_loss_history, val_loss_history, save_path=None, log_scale=False):
  plt.figure(figsize=(10, 6))
  
  plt.plot(epoch_history, train_loss_history, label='Train Loss', marker='o')
  plt.plot(epoch_history, val_loss_history, label='Validation Loss', marker='x')
  
  if log_scale:
    plt.yscale("log")
  
  plt.text(epoch_history[-1], val_loss_history[-1], f"{val_loss_history[-1]:.4f}", fontsize=9)
  
  plt.xlabel('Epoch')
  plt.ylabel('Loss (MSE / MAE / MAPE)')
  plt.title(f'Loss Progression up to Epoch {epoch_history[-1]}')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  
  if save_path is not None:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
