import os
import json
import torch
import argparse
import numpy as np
import PIL
from PIL import Image
import scipy.sparse as sp
import requests
from models import GraphRecommender
from feature import FeatureExtractor
from utils import expand_csr_adj


parser = argparse.ArgumentParser()
parser.add_argument('--image_url', type=str)
parser.add_argument('--category', required=True, type=str)
parser.add_argument('--image_path', type=str)
parser.add_argument('--k', default=1, type=int)
parser.add_argument('--topk', default=10, type=int)
args = parser.parse_args()

category = args.category
k = 0
topk=20
image_url =args.image_url



MODEL_DIR = 'data/models'
DATA_DIR = 'data'

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
recommender = GraphRecommender(model_dir=MODEL_DIR)
extractor = FeatureExtractor(device=device)

# load precomputed data
features = np.array(sp.load_npz(f"{DATA_DIR}/features_product.npz").todense())
with open(f"{DATA_DIR}/idx2id_product.json",  encoding='cp949') as f:
    idx2id = json.load(f)
with open(f"product_images/db_product_styling.json") as f2:
    products = json.load(f2)
product_category = {}
product_imageUrl = {}
for product in products:
    product_category[product['id']] = product['category']
    product_imageUrl[product['id']] = product['imageUrl_rm']

# read image
image = Image.open(requests.get(args.image_url, stream=True).raw)
image = image.convert('RGB')
image = np.array(image)

new_index = features.shape[0]

# append feature
feature = extractor.get_feature(image)
_features = np.vstack((features, feature))

if args.k == 0:
    output = recommender.predict_no_adj(
        query_index=new_index, features=_features, topk=args.topk)
else:
    _adj = expand_csr_adj(adj, count=1)
    output = recommender.predict(
        query_index=new_index, adj=_adj, features=_features, k=args.k, topk=args.topk)
i = 1
for index, score in output:
    print(i,  product_category[idx2id[str(index)]], idx2id[str(index)], score)
    i+=1

styling_product = []
if category == 'top':
    target_category = ['bottom', 'shoes']
    top_url = image_url
elif category == 'bottom':
    target_category = ['top', 'shoes']
    bottom_url = image_url
elif category == 'shoes':
    target_category = ['top', 'bottom']
    shoes_url = image_url
else:
    target_category = []

print(target_category)
for index, score in output:
    if len(styling_product) == len(target_category):
        break
    if product_category[idx2id[str(index)]] in target_category:
        styling_product.append(idx2id[str(index)])

for product in styling_product:
    if product_category[product] == 'top':
        top_url = product_imageUrl[product]
    elif product_category[product] == 'bottom':
        bottom_url = product_imageUrl[product]
    elif product_category[product] == 'shoes':
        shoes_url = product_imageUrl[product]
print(styling_product)
_image = Image.open(requests.get(top_url, stream=True).raw)
_image1 = Image.open(requests.get(bottom_url, stream=True).raw)
_image2 = Image.open(requests.get(shoes_url, stream=True).raw)

image = Image.new("RGBA", _image.size, "WHITE") # Create a white rgba background
image.paste(_image, (0, 0), _image)
image1 = Image.new("RGBA", _image1.size, "WHITE") # Create a white rgba background
image1.paste(_image1, (0, 0), _image1)
image2 = Image.new("RGBA", _image2.size, "WHITE") # Create a white rgba background
image2.paste(_image2, (0, 0), _image2)


#image2 = cv2.imdecode(image2_nparray, cv2.IMREAD_COLOR)
image_width = image.width
image_height = image.height
image1_width = image1.width
image1_height = image1.height
image2_width = image2.width
image2_height = image2.height

width = max([image_width, image1_width])#, image2_size[0]
height = max([image_height, image1_height])#, image2_size[1]
background = Image.new('RGBA', (width, image_height+image1_height+image2_height), "WHITE")


background.paste(image, (int(width/2-image_width/2), 0))
background.paste(image1, (int(width/2-image1_width/2), int(image_height)))
background.paste(image2, (int(width/2-image2_width/2), int(image_height+image1_height)))

#background.show()
#background.save("styling_image.jpg")
background.convert('RGB').save('styling_image.jpg', 'JPEG')