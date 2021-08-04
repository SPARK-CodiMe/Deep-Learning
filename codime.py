import os
import json
import torch
import argparse
import numpy as np
import PIL
from PIL import Image
import scipy.sparse as sp

from models import GraphRecommender
from feature import FeatureExtractor
from utils import expand_csr_adj


parser = argparse.ArgumentParser()
parser.add_argument('--image_url', required=True, type=str)
parser.add_argument('--k', default=1, type=int)
parser.add_argument('--topk', default=10, type=int)
args = parser.parse_args()


MODEL_DIR = 'data/models'
DATA_DIR = 'data'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
recommender = GraphRecommender(model_dir=MODEL_DIR)
extractor = FeatureExtractor(device=device)

# load precomputed data
features = np.array(sp.load_npz(f"{DATA_DIR}/features_product.npz").todense())
with open(f"{DATA_DIR}/idx2id_product.json") as f:
    idx2id = json.load(f)

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
    print(i, idx2id[str(index)], score)
    i+=1
