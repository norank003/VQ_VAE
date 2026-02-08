import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

class DataSetup:
  def __init__(self,batch_size:int,num_workers:int):
    self.batch_size=batch_size
    self.num_workers=num_workers
    self.train_data= torchvision.datasets.MNIST(root='data',
                                       train=True,
                                       download=True,

                                       transform=torchvision.transforms.ToTensor(),
                                       target_transform=None)

    self.test_data= torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          download=True,
                                          transform=torchvision.transforms.ToTensor())

  def get_train_dataloader(self):


    return (DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True),
            DataLoader(self.test_data,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True))
