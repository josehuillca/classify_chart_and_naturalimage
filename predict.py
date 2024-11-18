import torch, os, shutil, cv2
from tqdm import tqdm
from typing import List
from classification.nn.model import ImageClassifier
from classification.model.module import ClassificationModule
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


IMAGE_SIZE = (224,224)
CLASS_NAMES = ['chart', 'natural'] # Ordem correto
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
custom_image_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE,interpolation=InterpolationMode.BICUBIC),
            #normalize
        ])
device = "cuda" if torch.cuda.is_available() else "cpu"


def run_inference(checkpoint_filepath:str, list_imgs_path:List[str]):
    # Load model
    checkpoint = torch.load(checkpoint_filepath)

    net = ImageClassifier()
    model = ClassificationModule(net, **checkpoint['hyper_parameters'])
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(device)
    model.eval()
    res = list()
    with torch.inference_mode():
        for img in tqdm(list_imgs_path):
            # Load in custom image and convert the tensor values to float32
            custom_image = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB).type(torch.float32)

            # Divide the image pixel values by 255 to get them between [0, 1]
            custom_image = custom_image / 255. 

            # Transform target image
            custom_image_transformed = custom_image_transform(custom_image)

            # Make a prediction on image with an extra dimension
            custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(device))
            custom_image_pred_label = torch.argmax(torch.softmax(custom_image_pred, dim=1), dim=1)

            custom_image_pred_class = CLASS_NAMES[custom_image_pred_label.cpu()]
            res.append(custom_image_pred_class)
    #print(f"acc: {sum(res)/len(list_imgs_path)}")
    return res


def inference(model, img_path:str):
    with torch.inference_mode():
        # Load in custom image and convert the tensor values to float32
        custom_image = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB).type(torch.float32)

        # Divide the image pixel values by 255 to get them between [0, 1]
        custom_image = custom_image / 255. 

        # Transform target image
        custom_image_transformed = custom_image_transform(custom_image)

        # Make a prediction on image with an extra dimension
        custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(device))
        custom_image_pred_label = torch.argmax(torch.softmax(custom_image_pred, dim=1), dim=1)

        custom_image_pred_class = CLASS_NAMES[custom_image_pred_label.cpu()]
    return custom_image_pred_class


if __name__=='__main__':
    checkpoint_filepath = os.path.join('Classifier-naturalImages', '4ljh5lms', 'checkpoints', 'epoch=12-step=2353.ckpt') #

    path_base = os.path.join('dataset','naturalchart','test_set','natural')
    img_files = [os.path.join(path_base,img) for img in os.listdir(path_base)]
    res = run_inference(checkpoint_filepath, img_files)

    print('Finish..')