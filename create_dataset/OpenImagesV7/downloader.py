import os, tqdm, pandas as pd
import numpy as np
import os.path as op
import requests

def download_image(image_url, file_dir):
    response = requests.get(image_url)

    if response.status_code == 200:
        directory = os.path.dirname(file_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_dir, "wb") as fp:
            fp.write(response.content)
        print("Image downloaded successfully.")
    else:
        print(f"Failed to download the image. Status code: {response.status_code}")


def download(file_images:str, path_save:str, num_images_to_download:int):
    df = pd.read_csv(file_images,sep=',',header=0)
    images = df.iloc[:, :1].values.tolist()
    idxs = [i for i in range(1,len(images))]
    np.random.shuffle(idxs)
    
    if not op.exists(path_save): os.makedirs(path_save,exist_ok=True)

    list_images = [images[idx][0].replace('\t','?') for idx in idxs[:num_images_to_download]]

    idx = 0
    for image_url in tqdm.tqdm(list_images):
      img_name = f'image_{idx}.jpg'
      download_image(image_url, op.join(path_save, img_name))


# Dataset name: https://storage.googleapis.com/openimages/web/download_v7.html
# Dataset : https://github.com/cvdfoundation/open-images-dataset?tab=readme-ov-file#download-images-with-bounding-boxes-annotations
if __name__ == '__main__':
    np.random.seed(123)

    num_images_to_download = 5600
    path_save = '../../dataset/OpenImagesV7'
    images_file = 'open-images-dataset-test.tsv'
    download(images_file, path_save, num_images_to_download)

    print("finish..")
