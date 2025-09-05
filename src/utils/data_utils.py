import os
import torch 
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from PIL import Image

from typing import Optional

from transformers import AutoImageProcessor
class BoilerplateDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', processor: Optional[AutoImageProcessor] = None, is_training: bool = True):
        """
        Boilerplate Dataset for folder-based structure
        
        Args:
            root_dir: Path to the dataset root directory
            split: 'train', 'test', or 'valid'
            processor: SigLIP AutoImageProcessor for preprocessing
            is_training: Whether to apply training augmentations
        """
        self.root_dir = root_dir
        self.split = split
        self.processor = processor
        self.is_training = is_training
        
        self.split_dir = os.path.join(root_dir, split)
        
        self.classes = sorted([d for d in os.listdir(self.split_dir) 
                              if os.path.isdir(os.path.join(self.split_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        self.samples = []
        self._load_samples()
    
        self.augmentations = None
    
    def _load_samples(self):
        """Load all image paths and their corresponding labels"""
        for class_name in self.classes:
            class_dir = os.path.join(self.split_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    img_path = os.path.join(class_dir, filename)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.augmentations is not None:
            image = self.augmentations(image)
        
        if self.processor is not None:
            processed = self.processor(images=image, return_tensors="pt")
            pixel_values = processed['pixel_values'].squeeze(0)  
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)
    
    def get_class_names(self) -> List[str]:
        """Return list of class names"""
        return self.classes
    
    def get_class_to_idx(self) -> Dict[str, int]:
        """Return class name to index mapping"""
        return self.class_to_idx

    def get_class_names(self):
        return self.classes

class BoilerplateDataModule(pl.LightningDataModule):
  def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers

    model_str = 'google/siglip-base-patch16-224'
    self.processor = AutoImageProcessor.from_pretrained(model_str)
    

  def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = BoilerplateDataset(
                root_dir=self.data_dir,
                split='train',
                processor=self.processor,
                is_training=True
            )
            self.val_dataset = BoilerplateDataset(
                root_dir=self.data_dir,
                split='valid',
                processor=self.processor,
                is_training=False
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = BoilerplateDataset(
                root_dir=self.data_dir,
                split='test',
                processor=self.processor,
                is_training=False
            )
  
  def train_dataloader(self):
      return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=self.num_workers,
          pin_memory=True
      )
  
  def val_dataloader(self):
      return DataLoader(
          self.val_dataset,
          batch_size=self.batch_size,
          shuffle=False,
          num_workers=self.num_workers,
          pin_memory=True
      )
  
  def test_dataloader(self):
      return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          shuffle=False,
          num_workers=self.num_workers,
          pin_memory=True
      )


