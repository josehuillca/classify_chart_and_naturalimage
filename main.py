import torch
import os
import random
from PIL import Image
import glob
from pathlib import Path
from classification.nn.model import ImageClassifier
from classification.model.module import train

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


##### Loading Dataset
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
train_dir = "dataset/catdog/training_set"
test_dir = "dataset/catdog/test_set"

# understanding the dataset
def undertanding_dataset(image_path):
    # Set seed
    random.seed(42) 

    # 1. Get all image paths (* means "any combination")
    image_path_list= glob.glob(f"{image_path}/*/*/*.jpg")

    # 2. Get random image path
    random_image_path = random.choice(image_path_list)

    # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
    image_class = Path(random_image_path).parent.stem

    # 4. Open image
    img = Image.open(random_image_path)

    # 5. Print metadata
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}") 
    print(f"Image width: {img.width}")
    return image_path_list


###### Transforming data
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Set image size.
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

# Create training transform with TrivialAugment
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
            fig.suptitle(f"Class: {Path(image_path).parent.stem}", fontsize=16)
            plt.show() # debug


import torchvision
## Prediction model
def prediction(model,custom_image_path,class_names):
    # Load in custom image and convert the tensor values to float32
    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
    # Divide the image pixel values by 255 to get them between [0, 1]
    custom_image = custom_image / 255. 

    custom_image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
    ])

    # Transform target image
    custom_image_transformed = custom_image_transform(custom_image)

    # 3. Perform a forward pass on a single image
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to image
        custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
        # Make a prediction on image with an extra dimension
        pred = model(custom_image_transformed_with_batch_size.to(device))
        
    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    custom_image_pred_label = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    print(f"Output prediction label:\n{custom_image_pred_label}\n")
    custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label
    print(f"Predicts label:\n{custom_image_pred_class}")

def understanding_model():
    # Install torchinfo if it's not available, import it if it is
    try: 
        import torchinfo
    except:
        print("install: pip install torchinfo")
        
    from torchinfo import summary
    # do a test pass through of an example input size 
    print(summary(model, input_size=[1, 3, IMAGE_WIDTH ,IMAGE_HEIGHT]))


def plot_loss_curves(results):
    results = dict(list(model_results.items()))

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


if __name__=="__main__":
    image_path ="dataset/catdog"

    #image_path_list = undertanding_dataset(image_path)
    #plot_transformed_images(image_path_list, transform=data_transform, n=1)

    # Creating training set
    train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data_augmented = datasets.ImageFolder(test_dir, transform=test_transform)

    # Get class names as a list
    class_names = train_data_augmented.classes
    print("Class names: ",class_names)

    # Can also get class names as a dict
    class_dict = train_data_augmented.class_to_idx
    print("Class names as a dict: ",class_dict)

    # Check the lengths
    print("The lengths of the training and test sets: ", len(train_data_augmented), len(test_data_augmented))

    #Debug ====================
    img, label = train_data_augmented[0][0], train_data_augmented[0][1]
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")
    # Rearrange the order of dimensions
    img_permute = img.permute(1, 2, 0)
    # Print out different shapes (before and after permute)
    print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]") # TO matplotlib

    # Load datasets ==========================
    # How many subprocesses will be used for data loading (higher = more)
    NUM_WORKERS = 2 #os.cpu_count()

    # Turn train and test Datasets into DataLoaders
    # Set some parameters.
    BATCH_SIZE = 32
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    train_dataloader_augmented = DataLoader(train_data_augmented, 
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)

    test_dataloader_augmented = DataLoader(test_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=NUM_WORKERS)
    
    # Debug
    img, label = next(iter(train_data_augmented))
    # Note that batch size will now be 1.  
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label}")

    #Creating CNN Image Classifier
    # Instantiate an object.
    model = ImageClassifier().to(device)

    #understanding_model()

    # Train and Evaluate Model =======================================
    # Set number of epochs
    NUM_EPOCHS = 1

    # Setup loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model_0 
    model_results = train(model=model,
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_augmented,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS, device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

    plot_loss_curves(model_results)

    # Make a Prediction =======================
    prediction(model, custom_image_path='dataset/catdog/test_set/dogs/dog.4001.jpg', class_names=class_names)

    print("Finish...")