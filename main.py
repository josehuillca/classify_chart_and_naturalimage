import argparse, os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from classification.nn.model import ImageClassifier
from classification.model.module import ClassificationModule
from classification.data.datamodule import MyDataModule


def understanding_model(model, image_width, image_height):
    # Install torchinfo if it's not available, import it if it is
    try: 
        import torchinfo
    except:
        print("install: pip install torchinfo")
        
    from torchinfo import summary
    # do a test pass through of an example input size 
    print(summary(model, input_size=[1, 3, image_width, image_height]))


def main(args: argparse.Namespace) -> None:
    # Set image size.
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    net = ImageClassifier()
    model = ClassificationModule(net, lr=args.lr)
    datamodule = MyDataModule(args.dataset_root, image_size=IMAGE_SIZE, batch_size=args.batch_size, num_workers=args.num_workers)
    
    logger = WandbLogger(project='Classifier-naturalImages', log_model=True)

    trainer = Trainer(
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=os.path.join('.', 'checkpoints'),
        # gradient_clip_val=0.001,  # Veja na documentação do PyTorch Lightning o que isso faz.
        max_epochs=args.max_epochs,
        log_every_n_steps=min(50, 50),
        num_sanity_val_steps=0,  # Evita que testes de sanidade sobre os dados de validação sejam executados antes do treino.
    )

    if args.task == 'fit':
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.checkpoint)
    else:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=args.checkpoint)
    print("Class-names: ", datamodule.class_names)


if __name__=="__main__":
    # ---- Definir argumentos passados por linha de comando ao chamar o programa ----
    parser = argparse.ArgumentParser()
    # Definir tarefas disponíveis.
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fit', dest='task', action='store_const', const='fit', help='runs the full optimization routine')
    group.add_argument('--test', dest='task', action='store_const', const='test', help='perform one evaluation epoch over the test set')
    parser.set_defaults(task='fit')
    # Definir argumentos que configuram o dispositivo onde a tarefa será executada.
    group = parser.add_argument_group(title='Device')
    group.add_argument('--accelerator', metavar='TYPE', type=str, choices=['cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto'], default='auto', help='accelerator type (see PyTorch Lightning documentation)')
    group.add_argument('--devices', metavar='COUNT', type=int, default=1, help='number of devices to train (see PyTorch Lightning documentation)')
    # Definir argumentos relacionados com os dados utilizados.
    group = parser.add_argument_group(title='Data')
    group.add_argument('--batch_size', metavar='SIZE', type=int, default=32, help='size of the bach of images')
    group.add_argument('--num_workers', metavar='COUNT', type=int, default=2, help='number of workers used to load the image batches')
    group.add_argument('--dataset_root', metavar='PATH', type=str, default=os.path.join('.', 'dataset/catdog'), help='root dir for all datasets')
    # Definir argumentos relacionados com o modelo.
    group = group.add_argument_group(title='Model')
    group.add_argument('--lr', metavar='VALUE', type=float, default=1e-3, help='learning rate')
    group.add_argument('--max_epochs', metavar='COUNT', type=int, default=5, help='maximum number of epochs')
    group.add_argument('--checkpoint', metavar='PATH', type=str, default=None, help='path to some checkpoint (all other model arguments will be ignored)')
    # Chamar o método principal.
    main(parser.parse_args())