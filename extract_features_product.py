"""
Extract the features for each ssense image, using a resnet50 with pytorch
"""

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb
import skimage.io
import os
import argparse
from tqdm import tqdm
import pickle as pkl
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load net to extract features
model = models.resnet50(pretrained=True)
# skip last layer (the classifier)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(), normalize
            ])

def process_image(im):
    im = transform(im)
    im = im.unsqueeze_(0)
    im = im.to(device)
    out = model(im)
    return out.squeeze()

dataset_path = './product_images'
images_dir = os.path.join(dataset_path)
json_file = os.path.join(dataset_path, f'db_product.json')
with open(json_file) as f:
    data = json.load(f)

save_to = './'

#if not os.path.exists(save_to):
#    os.makedirs(save_to)
save_dict=os.path.join(save_to, f'imgs_featdict_product.pkl')

ids = {}

features = {}
count = {}

print('iterating through ids')
n_items = len(ids.keys())
with torch.no_grad():
    
    for product in tqdm(data):
        # Load clean image first
        if product['id'] in features:
            continue
        
        image_path = dataset_path + '/{}.png'.format(product['id'])
        print(image_path)
        
        if os.path.exists(image_path):
            try:
                img = skimage.io.imread(image_path)
            except Exception as e:
                print(e)
                continue
        else:
            continue
        
        img = np.array(img)
        if len(img.shape) == 2:
            img = gray2rgb(img)
        if img.shape[2] == 4:
            img = rgba2rgb(img)

        img = resize(img, (256, 256))
        img = img_as_ubyte(img)

        feats = process_image(img).cpu().numpy()

        features[product['id']] = feats

print(f'Total: {len(features)}')

with open(save_dict, 'wb') as handle:
    pkl.dump(features, handle, protocol=pkl.HIGHEST_PROTOCOL)
