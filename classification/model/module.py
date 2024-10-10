from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Tuple, Union
import operator
import torch, numpy as np
from ..typing_ import ImageBatch, Logits, Targets


class ClassificationModule(LightningModule):

    def __init__(self, net: Module, *, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['net'])  # Ignoramos o hyperparâmetro `net` por se tratar da instância da rede e não de valores simples escolhidos para treino ou configuração da rede.
        self.net = net
        self.lr = lr
        self.criterion = CrossEntropyLoss()
        self.step_outputs = {'Train':[],'Val':[],'Test':[]}

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

    def _common_step(self, batch: Tuple[ImageBatch, Targets], stage: str) -> Dict[str, Tensor]:
        # Carregar os dados de treino (input, target).
        images, target = batch  # images.shape = (batch_size, 3, width, height), target.shape = (batch_size,)
        # Processar os dados de entrada utilizando o modelo (net).
        logits = self.net(images)  # logits.shape = 
        
        # Utilizar o critério (loss function) para avaliar as saídas do modelo (logits), comparando-o com as anotações (target).
        loss = self.criterion(logits, target)

        # Calculate and accumulate accuracy metric across all batches
        pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        acc = (pred_class == target).sum().item()/len(pred_class)

        self.step_outputs[stage].append({'loss': loss, 'accuracy': acc})
        # Retornar loss e métricas calculadas
        return {'loss': loss, 'accuracy': acc}
    
    def _common_epoch_end(self, stage: str) -> None:
        # O conjunto de dados foi subdividido em vários batches e por isso o resultado precisa ser consolidado.
        loss = torch.stack(list(map(operator.itemgetter('loss'), self.step_outputs[stage]))).mean()
        accuracy = np.mean(list(map(operator.itemgetter('accuracy'), self.step_outputs[stage])))
        # Fazer log dos valores de interesse.
        self.log(f'Loss/{stage}', loss.detach().cpu())
        self.log(f'Accuracy/{stage}', accuracy)
        # free up the memory
        self.step_outputs[stage].clear()


    def training_step(self, batch, batch_idx: int):
        return self._common_step(batch, "Train")

    def on_train_epoch_end(self) -> None:
        self._common_epoch_end("Train")


    def validation_step(self, batch, batch_idx: int) -> None:
        return self._common_step(batch, "Val")

    def on_validation_epoch_end(self) -> None:
        self._common_epoch_end("Val")

    def test_step(self, batch, batch_idx: int) -> None:
        self._common_step(batch, "Test")
