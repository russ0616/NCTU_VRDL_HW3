from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform
from utils.functions import SavePath
from layers.output_utils import postprocess
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data import cfg, set_cfg

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import random
import json
import os
import time
import datetime
from pathlib import Path
from PIL import Image
import glob

import matplotlib.pyplot as plt
import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')

    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--score_threshold', default=0.15, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--trained_model',
                        default='weights/yolact_base_90_5460.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    
    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

class Detections:

    def __init__(self):
        self.mask_data = []

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings
        
        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': int(category_id)+1,
            'segmentation': rle,
            'score': float(score)
        })
    
    def dump(self):
        with open(args.mask_det_file, 'w') as f:
            json.dump(self.mask_data, f)        

def evalimage(net:Yolact, path:str):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))

    pre_time = time.time()
    preds = net(batch)
    aft_time = time.time()
    timee = 0
    timee = aft_time - pre_time
    
    return preds, timee
    
def evalimages(net:Yolact):
    times = 0
    cocoGt = COCO("data/test.json")
    detections = Detections()
    for imgid in cocoGt.imgs:
        name = cocoGt.loadImgs(ids=imgid)[0]['file_name']
        h = cocoGt.loadImgs(ids=imgid)[0]['height']
        w = cocoGt.loadImgs(ids=imgid)[0]['width']
        path = "data/test_images/" + name
        predict, timee = evalimage(net, path)
        times += timee
        classes, scores, boxes, masks = postprocess(predict, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)
        
        classes = list(classes.cpu().numpy().astype(int))
        scores = list(scores.cpu().numpy().astype(float))  
        print(scores)
        masks = masks.cpu().numpy()   
        boxes = boxes.cpu().numpy()    
                
        if len(classes) > 0: # If any objects are detected in this image
            for i in range(masks.shape[0]): # Loop all instances
                # save information of the instance in a dictionary then append on coco_dt list
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_mask(imgid, classes[i], masks[i,:,:], scores[i])

    detections.dump()
    print(times/100)
    print('Done.')

def evaluate(net:Yolact):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    evalimages(net)
    return

if __name__ == '__main__':
    parse_args()    
    
    model_path = SavePath.from_str(args.trained_model)
    # args.config = model_path.model_name + '_config'
    args.config = 'yolact_base_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
    set_cfg(args.config)        

    with torch.no_grad():        

        if args.cuda:            
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')       

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)        
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net)


