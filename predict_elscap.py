import os, glob, json, torch, cv2
import os.path as op
from tqdm import tqdm
from typing_ import *
from predict import device, inference
from classification.nn.model import ImageClassifier
from classification.model.module import ClassificationModule


ROOT = '/mnt/Mango-SSD/Mango2024/GitHub/elsevier_crawler'
ROOT_DST = '/home/hillai360/Documentos/Mango/elseivier'


def get_primary_level_list():
    base_path = op.join('dataset', 'images_info')
    return base_path, [plvl for plvl in os.listdir(op.join(ROOT, base_path))]


def get_jsonfile_from_path(path:str):
    return glob.glob(op.join(path, '**/*.json'), recursive = True)


def inference_naturalchart(model, list_jsonfile, dst_path:str):
    for file in tqdm(list_jsonfile, desc="      Inference"):
        with open(file) as f:
            data = json.load(f)
        img_path = op.join(op.dirname(file), op.basename(data['image']['file']))

        if not op.exists(img_path): # Algumas imagens nao foram baixadas
            print(f"NOT eXISTS: {img_path}")
            continue
        if img_path[-4:]=='.gif': # acontece pouco -- problemas com GIF images
            cap = cv2.VideoCapture(img_path)
            ret, image = cap.read()
            if ret:
                img_path = op.join('dataset','tests', op.basename(data['image']['file'])[:-4]+'.jpg')
                cv2.imwrite(img_path, image)
            else: continue
        
        try: # algumas imagens foram baixadas de forma incorreta
            figure_type=inference(model, img_path)
        except:
            continue

        new_dict = {
            'paper': Paper(
                prism_doi=data['paper']['prism_doi'],
                pii=data['paper']['pii'],
                openaccess=data['paper']['openaccess'],
                xml=op.join('dataset', data['paper']['xml'].split('dataset/')[1])
            ),
            'figure': Figure(
                refid=data['figure']['refid'],
                caption=data['figure']['caption'],
                figure_type=figure_type,
                chart_type=None,
                contain_subfigure=None,
                bbox_subfigure=None
            ),
            'image': Image(
                ref=data['image']['ref'],
                category=data['image']['category'],
                type=data['image']['type'],
                mimetype=data['image']['mimetype'],
                width=data['image']['width'],
                height=data['image']['height'],
                url=data['image']['url'],
                file=op.join('dataset', data['image']['file'].split('dataset/')[1])
            ),
            'paragraphs': [Paragraph(id=p['id'],text=p['text']) for p in data['paragraphs']['paragraphs']]
        }
        # 
        path_out = op.join(ROOT_DST, op.dirname(op.join('dataset', data['image']['file'].split('dataset/')[1])))
        if not op.exists(path_out): os.makedirs(path_out, exist_ok=True)
        with open(op.join(path_out, op.basename(file)), 'w', encoding='utf-8') as fout:
            json.dump(new_dict, fout, ensure_ascii=False)
        pass


if __name__=='__main__':
    checkpoint_filepath = op.join('Classifier-naturalImages', '4ljh5lms', 'checkpoints', 'epoch=12-step=2353.ckpt')
    # Load model
    checkpoint = torch.load(checkpoint_filepath)

    net = ImageClassifier()
    model_naturalchart = ClassificationModule(net, **checkpoint['hyper_parameters'])
    model_naturalchart.load_state_dict(checkpoint['state_dict'])
    model_naturalchart = model_naturalchart.to(device)
    model_naturalchart.eval()

    dst_path = ''

    base_path, p_lvl = get_primary_level_list()
    for name in tqdm(p_lvl, desc="  Primary-lvl"):
        list_json = get_jsonfile_from_path(op.join(ROOT,base_path,name))
        inference_naturalchart(model_naturalchart, list_json, dst_path)
    print('Finish', p_lvl)