from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomRotation, RandomAdjustSharpness
from PIL import Image
import torch
import numpy as np 

class TorchDataset(Dataset):
  def __init__(self, hf_dataset, processor=None, is_training=False):

    self.hf_dataset = hf_dataset
    self.processor = processor
    self.is_training = is_training
    
    if is_training:
      self.augmentations = [
        RandomRotation(degrees=15),
        RandomAdjustSharpness(sharpness_factor=1.5)
      ]
    else:
      self.augmentations = []

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

    if self.is_training:
      for aug in self.augmentations:
        if np.random.random() < 0.5: 
          image = aug(image)
    
    if self.processor:
      inputs = self.processor(image, return_tensors="pt")
      image = inputs['pixel_values'].squeeze(0) 

    return image, label  
  
class StanfordCarsDataset:
  def __init__(self, processor=None, is_training=False):
    self.processor = processor
    self.is_training = is_training

  def load_data(self):
    dataset = load_dataset("tanganke/stanford_cars")
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    return train_dataset, test_dataset
  
  def to_torch_dataset(self, hf_dataset):
    return TorchDataset(hf_dataset, self.processor, self.is_training)
  
