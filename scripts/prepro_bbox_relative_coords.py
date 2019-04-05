"""
Preprocess the absolute bounding box coordinates from --input_box_dir,
To convert these into relative coordinates, this script loads the corresponding images from coco/val2014/ and coco/train2014/, to get (img_width, img_height)

Input:
input_box_dir="/mydisk/Data/captioning_data/cocobu_adaptive_box"
info_filepath="/mydisk/Data/captioning_data/dataset_coco.json"
img_dir      ="/mydisk/Data/captioning_data/coco"

Output:
A directory containing all the boxes relative coordinates, as npy files.
"""

import os
import os.path as pth
from glob import glob
import re
import json
import requests
from io import BytesIO
import numpy as np
import PIL.Image
import argparse


def _pil_to_nparray(pim):
    image = pim.convert("RGB")
    imageArray=np.array(image)
    return (imageArray)

def get_numpy_image(url_or_filepath):
    """
    Converts an image url or filepath to its numpy array.

    :params str url_or_filepath:
        the url or filepath of the image we want to convert
    :returns np.array:
        'RGB' np.array representing the image
    """
    #image_url
    if 'http' in url_or_filepath or 'www' in url_or_filepath:
        url = url_or_filepath
        response = requests.get(url)
        pim = PIL.Image.open(BytesIO(response.content))
    #image_file
    else:
        filepath = url_or_filepath
        pim = PIL.Image.open(filepath)

    nim = _pil_to_nparray(pim)
    return nim

def get_bbox_relative_coords(params):
    input_box_dir=params['input_box_dir']
    info_filepath=params['input_json']
    img_dir      =params['image_root']
    output_dir   =params['output_dir']

    print("Reading coco dataset info")
    with open(info_filepath, "rb") as infile:
        coco_dict =json.load(infile)
    coco_ids_to_paths={str(img['cocoid']):os.path.join(img_dir,img['filepath'],img['filename']) for img in coco_dict['images'] }

    rel_box_dir=output_dir
    print("Output dir: %s"%rel_box_dir)
    if not os.path.exists(rel_box_dir):
        os.makedirs(rel_box_dir)


    box_files=sorted(glob(pth.join(input_box_dir,'*')))
    for ind, box_file in enumerate(box_files):
        if ind%1000==0:
            print ('processed %d images (of %d)' %(ind, len(box_files)))
        filenumber=box_file.split("/")[-1].split('.')[0]
        img_path=coco_ids_to_paths[filenumber]
        img_array=get_numpy_image(img_path)
        height, width = img_array.shape[:2]
        box=np.load(box_file)
        relative_box= box/np.array([width, height, width,height])
        relative_box=np.clip(relative_box,0.0,1.0)
        new_filename = pth.join(output_dir,filenumber+'.npy')
        np.save(new_filename, relative_box)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default='/mydisk/Data/captioning_data/dataset_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--image_root', type=str, default='/mydisk/Data/captioning_data/coco',
                    help='In case the image paths have to be preprended with a root path to an image folder')
    parser.add_argument('--input_box_dir', type=str, default='/mydisk/Data/captioning_data/cocobu_adaptive_box',
                    help='path to the directory containing the boxes of att feats')
    parser.add_argument('--output_dir', type=str, default='/mydisk/Data/captioning_data/zcocobu_adaptive_box_relative',
                    help='directory containing the files with relative coordinates of the bboxes in --input_box_dir')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    get_bbox_relative_coords(params)
