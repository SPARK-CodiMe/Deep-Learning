from flask import Flask
from flask import request, send_file
import os
import json
import torch
import argparse
import numpy as np
import io
from PIL import Image
import scipy.sparse as sp
import requests
from base64 import encodebytes
from models import GraphRecommender
from feature import FeatureExtractor
from utils import expand_csr_adj
import shutil
from get_result import getResult
import subprocess
app = Flask(__name__)

@app.route('/styling', methods=['POST'])
def get_result():
    image_url = request.form['image_url']
    category = request.form['category']
    codi_id = request.form['codi_id']

    return subprocess.call('python example.py --image_url {} --category {} --k 0 --topk 30'.format(image_url, category), shell=True)
    #return getResult(image_url, category, codi_id)

    # k=0
    # topk=50


    # MODEL_DIR = 'data/models'
    # DATA_DIR = 'data'

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # recommender = GraphRecommender(model_dir=MODEL_DIR)
    # extractor = FeatureExtractor(device=device)

    # # load precomputed data
    # features = np.array(sp.load_npz(f"{DATA_DIR}/features_product.npz").todense())
    # with open(f"{DATA_DIR}/idx2id_product.json") as f:
    #     idx2id = json.load(f)
    # with open(f"product_images/db_product_styling.json") as f2:
    #     products = json.load(f2)
    # product_category = {}
    # product_imageUrl = {}
    # for product in products:
    #     product_category[product['id']] = product['category']
    #     product_imageUrl[product['id']] = product['imageUrl_rm']

    # # read image
    # image = Image.open(requests.get(image_url, stream=True).raw)
    # image = image.convert('RGB')
    # image = np.array(image)

    # new_index = features.shape[0]

    # # append feature
    # feature = extractor.get_feature(image)
    # _features = np.vstack((features, feature))

    # if k == 0:
    #     output = recommender.predict_no_adj(
    #         query_index=new_index, features=_features, topk=topk)
    # else:
    #     _adj = expand_csr_adj(adj, count=1)
    #     output = recommender.predict(
    #         query_index=new_index, adj=_adj, features=_features, k=k, topk=topk)
    # i = 1
    # for index, score in output:
    #     print(i,  product_category[idx2id[str(index)]],idx2id[str(index)], score)
    #     i+=1

    # products = []
    # product1 = ''
    # product2 = ''
    # if category == 'top':
    #     target_category = ['bottom', 'shoes']
    #     top_url = image_url
    #     bottom_url = ''
    #     shoes_url = ''
        
    #     for index, score in output:
    #         if bottom_url != '' and shoes_url != '':
    #             break
    #         if product_category[idx2id[str(index)]] == 'bottom':
    #             bottom_url = product_imageUrl[idx2id[str(index)]]
    #             product1 = idx2id[str(index)]
    #         if product_category[idx2id[str(index)]] == 'shoes':
    #             shoes_url = product_imageUrl[idx2id[str(index)]]
    #             product2= idx2id[str(index)]
    #     if bottom_url != '' and shoes_url != '':
    #         products.append(product1)
    #         products.append(product2)
    #         _image = Image.open(requests.get(top_url, stream=True).raw).convert("RGBA")
    #         _image1 = Image.open(requests.get(bottom_url, stream=True).raw).convert("RGBA")
    #         _image2 = Image.open(requests.get(shoes_url, stream=True).raw).convert("RGBA")

    #         image = Image.new("RGBA", _image.size, "WHITE") # Create a white rgba background
    #         image.paste(_image, (0, 0), _image)
    #         image1 = Image.new("RGBA", _image1.size, "WHITE") # Create a white rgba background
    #         image1.paste(_image1, (0, 0), _image1)
    #         image2 = Image.new("RGBA", _image2.size, "WHITE") # Create a white rgba background
    #         image2.paste(_image2, (0, 0), _image2)

    #         image_width = image.width
    #         image_height = image.height
    #         image1_width = image1.width
    #         image1_height = image1.height
    #         image2_width = image2.width
    #         image2_height = image2.height

    #         width = max([image_width, image1_width, image2_width])
    #         background = Image.new('RGBA', (width, image_height+image1_height+image2_height), "WHITE")
    #         background.paste(image, (int(width/2-image_width/2), 0))
    #         background.paste(image1, (int(width/2-image1_width/2), int(image_height)))
    #         background.paste(image2, (int(width/2-image2_width/2), int(image_height+image1_height)))
    #         background.convert('RGB').save('codiImage/{}.jpg'.format(codi_id), 'JPEG')
    #         pil_img = Image.open('codiImage/{}.jpg'.format(codi_id), mode='r')
    #         byte_arr = io.BytesIO()
    #         pil_img.save(byte_arr, format='JPEG')
    #         encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    #         return { 
    #             'file': encoded_img,
    #             'products': products
    #          }
    #     elif bottom_url != '' and shoes_url =='':
    #         products.append(product1)
    #         _image = Image.open(requests.get(top_url, stream=True).raw).convert("RGBA")
    #         _image1 = Image.open(requests.get(bottom_url, stream=True).raw).convert("RGBA")
            

    #         image = Image.new("RGBA", _image.size, "WHITE") # Create a white rgba background
    #         image.paste(_image, (0, 0), _image)
    #         image1 = Image.new("RGBA", _image1.size, "WHITE") # Create a white rgba background
    #         image1.paste(_image1, (0, 0), _image1)

    #         image_width = image.width
    #         image_height = image.height
    #         image1_width = image1.width
    #         image1_height = image1.height

    #         width = max([image_width, image1_width])
    #         background = Image.new('RGBA', (width, image_height+image1_height), "WHITE")
    #         background.paste(image, (int(width/2-image_width/2), 0))
    #         background.paste(image1, (int(width/2-image1_width/2), int(image_height)))
    #         background.convert('RGB').save('codiImage/{}.jpg'.format(codi_id), 'JPEG')
    #         pil_img = Image.open('codiImage/{}.jpg'.format(codi_id), mode='r')
    #         byte_arr = io.BytesIO()
    #         pil_img.save(byte_arr, format='JPEG')
    #         encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    #         return { 
    #             'file': encoded_img,
    #             'products': products
    #          }
    #     else:
    #         return;
    # elif category == 'bottom':
    #     target_category = ['top', 'shoes']
    #     bottom_url = image_url
    #     top_url = ''
    #     shoes_url = ''
    #     for index, score in output:
    #         if top_url != '' and shoes_url != '':
    #             break
    #         if product_category[idx2id[str(index)]] == 'top':
    #             top_url = product_imageUrl[idx2id[str(index)]]
    #             product1 = idx2id[str(index)]
    #         if product_category[idx2id[str(index)]] == 'shoes':
    #             shoes_url = product_imageUrl[idx2id[str(index)]]
    #             product2 = idx2id[str(index)]
    #     if top_url != '' and shoes_url != '':
    #         products.append(product1)
    #         products.append(product2)
    #         _image = Image.open(requests.get(top_url, stream=True).raw).convert("RGBA")
    #         _image1 = Image.open(requests.get(bottom_url, stream=True).raw).convert("RGBA")
    #         _image2 = Image.open(requests.get(shoes_url, stream=True).raw).convert("RGBA")

    #         image = Image.new("RGBA", _image.size, "WHITE") # Create a white rgba background
    #         image.paste(_image, (0, 0), _image)
    #         image1 = Image.new("RGBA", _image1.size, "WHITE") # Create a white rgba background
    #         image1.paste(_image1, (0, 0), _image1)
    #         image2 = Image.new("RGBA", _image2.size, "WHITE") # Create a white rgba background
    #         image2.paste(_image2, (0, 0), _image2)

    #         image_width = image.width
    #         image_height = image.height
    #         image1_width = image1.width
    #         image1_height = image1.height
    #         image2_width = image2.width
    #         image2_height = image2.height

    #         width = max([image_width, image1_width, image2_width])
    #         background = Image.new('RGBA', (width, image_height+image1_height+image2_height), "WHITE")
    #         background.paste(image, (int(width/2-image_width/2), 0))
    #         background.paste(image1, (int(width/2-image1_width/2), int(image_height)))
    #         background.paste(image2, (int(width/2-image2_width/2), int(image_height+image1_height)))
    #         background.convert('RGB').save('codiImage/{}.jpg'.format(codi_id), 'JPEG')
    #         pil_img = Image.open('codiImage/{}.jpg'.format(codi_id), mode='r')
    #         byte_arr = io.BytesIO()
    #         pil_img.save(byte_arr, format='JPEG')
    #         encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    #         return { 
    #             'file': encoded_img,
    #             'products': products
    #          }
    #     elif top_url != '' and shoes_url =='':
    #         products.append(product1)
    #         _image = Image.open(requests.get(top_url, stream=True).raw).convert("RGBA")
    #         _image1 = Image.open(requests.get(bottom_url, stream=True).raw).convert("RGBA")
            

    #         image = Image.new("RGBA", _image.size, "WHITE") # Create a white rgba background
    #         image.paste(_image, (0, 0), _image)
    #         image1 = Image.new("RGBA", _image1.size, "WHITE") # Create a white rgba background
    #         image1.paste(_image1, (0, 0), _image1)

    #         image_width = image.width
    #         image_height = image.height
    #         image1_width = image1.width
    #         image1_height = image1.height

    #         width = max([image_width, image1_width])
    #         background = Image.new('RGBA', (width, image_height+image1_height), "WHITE")
    #         background.paste(image, (int(width/2-image_width/2), 0))
    #         background.paste(image1, (int(width/2-image1_width/2), int(image_height)))
    #         background.convert('RGB').save('codiImage/{}.jpg'.format(codi_id), 'JPEG')
    #         pil_img = Image.open('codiImage/{}.jpg'.format(codi_id), mode='r')
    #         byte_arr = io.BytesIO()
    #         pil_img.save(byte_arr, format='JPEG')
    #         encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    #         return { 
    #             'file': encoded_img,
    #             'products': products
    #          }
    #     else:
    #         return;
    # elif category == 'onepiece':
    #     target_category = ['shoes']
    #     onepiece_url = image_url
    #     shoes_url = ''
    #     for index, score in output:
    #         if shoes_url != '':
    #             break
    #         if product_category[idx2id[str(index)]] == 'shoes':
    #             shoes_url = product_imageUrl[idx2id[str(index)]]
    #             product1 = idx2id[str(index)]
    #     if  shoes_url != '':
    #         products.append(product1)
    #         _image = Image.open(requests.get(onepiece_url, stream=True).raw).convert("RGBA")
    #         _image2 = Image.open(requests.get(shoes_url, stream=True).raw).convert("RGBA")

    #         image = Image.new("RGBA", _image.size, "WHITE") # Create a white rgba background
    #         image.paste(_image, (0, 0), _image)
    #         image2 = Image.new("RGBA", _image2.size, "WHITE") # Create a white rgba background
    #         image2.paste(_image2, (0, 0), _image2)

    #         image_width = image.width
    #         image_height = image.height
    #         image2_width = image2.width
    #         image2_height = image2.height

    #         width = max([image_width, image2_width])
    #         background = Image.new('RGBA', (width, image_height+image2_height), "WHITE")
    #         background.paste(image, (int(width/2-image_width/2), 0))
    #         background.paste(image2, (int(width/2-image2_width/2), int(image_height)))
    #         background.convert('RGB').save('styling_image.jpg', 'JPEG')
    #         pil_img = Image.open('codiImage/{}.jpg'.format(codi_id), mode='r')
    #         byte_arr = io.BytesIO()
    #         pil_img.save(byte_arr, format='JPEG')
    #         encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    #         return { 
    #             'file': encoded_img,
    #             'products': products
    #          }
    #     else:
    #         return;
        
  

if __name__ == '__main__':
    app.run(host='14.49.45.139', port=8443)