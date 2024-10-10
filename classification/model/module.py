from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Tuple, Union
import operator
import torch, torchmetrics
from ..typing_ import ImageBatch, Logits, Targets


class ClassificationModule(LightningModule):

    def __init__(self, net: Module, *, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['net'])  # Ignoramos o hyperparâmetro `net` por se tratar da instância da rede e não de valores simples escolhidos para treino ou configuração da rede.
        self.net = net
        self.lr = lr
        self.criterion = CrossEntropyLoss()

    def configure_callbacks(self) -> List[Callback]:
        return [EarlyStopping(monitor='Accuracy/Val', mode='max', patience=6, strict=False)]  # O argumento strict=False é necessário para poder reiniciar o treinamento de onde parou anteriormente.

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'Accuracy/Val',
                'strict': False,  # O argumento strict=False é necessário para poder reiniciar o treinamento de onde parou anteriormente.
            }
        }
    
    def forward(self, images: ImageBatch) -> Logits:
        return self.net(images)

    def _common_step(self, batch: Tuple[ImageBatch, Targets]) -> Dict[str, Tensor]:
        pass

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


from tqdm.auto import tqdm
from torch import nn
# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device=None):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer, device=device)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn, device=device)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results