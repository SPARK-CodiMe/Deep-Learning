"""
상품 추천을 위한 전처리
"""

import os
import random
import json
import scipy as sp
from scipy.sparse import lil_matrix, save_npz, csr_matrix
import argparse
import pickle as pkl
import numpy as np



with open(f'product_images/product_list.json') as f:
    json_data = json.load(f)

# load the features extracted with 'extract_features.py'
feat_pkl = os.path.join(f'imgs_featdict_product.pkl')
if os.path.exists(feat_pkl):
    with open(feat_pkl, 'rb') as f:
        feat_dict = pkl.load(f)
else:
    raise FileNotFound('The extracted features file {} does not exist'.format(feat_pkl))


id2idx = {}
idx2id = {}
features = []

product_ids = set(product['id'] for product in json_data)
product_ids = [id for id in product_ids if id in feat_dict]

features_mat = np.zeros((len(product_ids), 2048))
print("product num: {}".format(len(product_ids)))

for i, id in enumerate(product_ids):
    idx2id[i] = id
    id2idx[id] = i
    features_mat[i] = feat_dict[id]


        
map_file1 = os.path.join(f"id2idx_product.json")
with open(map_file1, 'w') as f:
    json.dump(id2idx, f)

map_file2 = os.path.join(f"idx2id_product.json")
with open(map_file2, 'w') as f:
    json.dump(idx2id, f)


save_feat_file = os.path.join( f'features_product.npz')
sp_feat = csr_matrix(features_mat)
save_npz(save_feat_file, sp_feat)

