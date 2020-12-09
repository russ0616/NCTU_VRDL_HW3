import json
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

json_path = './pascal_train.json'
val_img_num = 149


train_imgs = []
val_anno = []
train_anno = []
train_json = {}
val_json = {}

with open(json_path) as json_file: 
    results = json.load(json_file)
    
# for i in range(len(results["annotations"])):
#     results["annotations"][i]["id"] = str(results["annotations"][i]["id"])
#     results["annotations"][i]["image_id"] = str(results["annotations"][i]["image_id"])
#     results["annotations"][i]["category_id"] = str(results["annotations"][i]["category_id"])
#     results["annotations"][i]["area"] = float(results["annotations"][i]["area"])
    
    
# for i in range(len(results["categories"])):
#     results["categories"][i]["id"] = str(results["categories"][i]["id"])
    
# for i in range(len(results["images"])):
#     results["images"][i]["id"] = str(results["images"][i]["id"])
    
val_imgs = random.sample(results["images"], val_img_num)

for img in results["images"]:
    flag = False
    for val_img in val_imgs:
        if img == val_img:
            flag = True
            break
        
    if flag:
        continue
    
    train_imgs.append(img)

for anno in results["annotations"]:
    anno_id = anno["image_id"]
    flag = True
    for val_img in val_imgs:
        if anno_id == val_img["id"]:
            val_anno.append(anno)
            flag = False
            break
        
    if flag:
        train_anno.append(anno)
        
train_json["annotations"] = train_anno
train_json["categories"] = results["categories"]
train_json["images"] = train_imgs

val_json["annotations"] = val_anno
val_json["categories"] = results["categories"]
val_json["images"] = val_imgs

with open("./pascal_sbd_train.json", 'w') as js:
    json.dump(train_json, js, ensure_ascii=False)
    
with open("./pascal_sbd_val.json", 'w') as js:
    json.dump(val_json, js, ensure_ascii=False)