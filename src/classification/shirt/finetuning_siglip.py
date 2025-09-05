from classification.cap.cap_dataset import CapDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomAdjustSharpness, RandomRotation

import numpy as np

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import  EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import (
    AutoImageProcessor, 
    Siglip2ForImageClassification
)
from .shirt_dataset import ShirtDataset 

class ImageClassificationModule(pl.LightningModule):
  def __init__(self, model_name: str, num_classes: int, learning_rate: float = 1e-4, weight_decay: float = 0.01, id2label=None, label2id=None):
    super().__init__()
    self.save_hyperparameters()
    self.model = Siglip2ForImageClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_classes=num_classes, ignore_mismatched_sizes=True,id2label=id2label, label2id=label2id)
    
    self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, x):
    out = self.model(x)
    return out.logits

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)

    return {
      'optimizer': optimizer,
      'lr_scheduler': {
        'scheduler': scheduler,
        'interval': 'epoch',
        'frequency': 1
      }
    }
   
class ShirtDataModule(pl.LightningDataModule):
  def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers

    model_str = 'google/siglip2-base-patch16-224'
    self.processor = AutoImageProcessor.from_pretrained(model_str)
    

  def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ShirtDataset(
                root_dir=self.data_dir,
                split='train',
                processor=self.processor,
                is_training=True
            )
            self.val_dataset = ShirtDataset(
                root_dir=self.data_dir,
                split='valid',
                processor=self.processor,
                is_training=False
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = ShirtDataset(
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

 

