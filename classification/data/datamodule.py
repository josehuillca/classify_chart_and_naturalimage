import os
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional, Tuple
from torchvision import datasets, transforms


class MyDataModule(LightningDataModule):

    def __init__(self, root: str, *, image_size: Tuple[int,int], batch_size: int, num_workers: int):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names = None


    def setup(self, stage: Optional[str] = None) -> None:
        # Create training transform with TrivialAugment
        train_transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.TrivialAugmentWide(),
                            transforms.ToTensor()])
        # Create testing transform (no data augmentation)
        test_transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.ToTensor()])
        if stage == 'fit' or stage is None: # TODO: dividir o trainset para val_subset
            self.train_subset = datasets.ImageFolder(os.path.join(self.root, 'training_set'), transform=train_transform)
            self.val_subset = datasets.ImageFolder(os.path.join(self.root, 'test_set'), transform=test_transform)
            self.class_names = self.val_subset.classes
            
        if stage == 'test' or stage is None: 
            self.test_subset = datasets.ImageFolder(os.path.join(self.root, 'test_set'), transform=test_transform)
            self.class_names = self.test_subset.classes

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False)