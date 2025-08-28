from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from PIL import Image
import torch 

class TorchDataset(Dataset):
  def __init__(self, hf_dataset, transform=None):

    self.hf_dataset = hf_dataset
    self.transform = transform 

  def __len__(self):
    return len(self.hf_dataset)
  
  def __getitem__(self, idx):
    item = self.hf_dataset[idx]
    image = item['image']
    
    if isinstance(image, str):
      image = Image.open(image)

    if image.mode != 'RGB':
      image = image.convert('RGB')

    label = item['label'] if 'label' in item else item['labels']

    if self.transform:
      image = self.transform(image)

    return image, label  
  
class StanfordCarsDataset:
  def __init__(self, transform=None):
    self.transform = transform

  def load_data(self):
    dataset = load_dataset("tanganke/stanford_cars")
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    return train_dataset, test_dataset
  
  def to_torch_dataset(self, hf_dataset):
    return TorchDataset(hf_dataset, self.transform)
  
