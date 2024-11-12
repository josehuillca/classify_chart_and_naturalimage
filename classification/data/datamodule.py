import os
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional, Tuple
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from .randaugment import RandomAugment


class MyDataModule(LightningDataModule):

    def __init__(self, root: str, *, image_size: Tuple[int,int], batch_size: int, num_workers: int):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names = None


    def setup(self, stage: Optional[str] = None) -> None:
        min_scale=0.5
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        """# Create training transform with TrivialAugment
        train_transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.TrivialAugmentWide(),
                            transforms.ToTensor()])
        # Create testing transform (no data augmentation)
        test_transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.ToTensor()])
        """
        train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(self.image_size,scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
        test_transform = transforms.Compose([
            transforms.Resize(self.image_size,interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ]) 
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