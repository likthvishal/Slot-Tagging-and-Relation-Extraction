# Create Dataset class
import torch
from torch.utils.data import Dataset, DataLoader

class MovieData(Dataset):
  def __init__(self, X, y):
    try:
        self.X = torch.tensor(X)
    except ValueError as e:
        self.X = [torch.tensor(i) for i in X]

    self.y = torch.tensor(y)
  
  def __len__(self):
    return len(self.y)

  def __getitem__(self,index):
    return self.X[index], self.y[index]