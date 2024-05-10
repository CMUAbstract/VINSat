import argparse
import os
import numpy as np
import rasterio as rio
from tqdm.contrib.concurrent import process_map
import pyproj
from PIL import Image
import cv2

def parse_args():
    """
    Parse command line arguments for labeling Data.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Labeling Data')
    parser.add_argument('--landmark_path', type=str, required=True, help='Path to landmark file')
    parser.add_argument('--raster_dir_path', type=str, required=True, help='Path to raster file')
    parser.add_argument('--output_dir_path', type=str, required=True, help='Path to save labels')
    parser.add_argument('-v', '--viz_labels', action='store_true', help='Visualize labels')
    parser.add_argument('-c', '--calculate_labels', action='store_true', help='Calculate labels')
    parsed_args = parser.parse_args()
    return parsed_args

def get_landmarks(landmark_path):
    """
    Load landmarks from a given file path.

    Parameters:
    landmark_path (str): The path to the landmark file.

    Returns:
    numpy.ndarray: The loaded landmarks.
    """
    loaded_landmarks = np.load(landmark_path)
    return loaded_landmarks

def get_raster_paths(raster_dir_path):
    """
    Get the paths of all raster files (with .tif extension) in the given directory and its subdirectories.
    
    Args:
        raster_dir_path (str): The path to the directory containing the raster files.
        
    Returns:
        list: A list of paths to the raster files.
    """
    raster_paths = []
    for root, dirs, files in os.walk(raster_dir_path):
        for file in files:
            if file.endswith('.tif'):
                raster_paths.append(os.path.join(root, file))
    return raster_paths

def label_raster(raster_path):
    with rio.open(raster_path) as src:
        crs = src.crs
        im_width = src.width
        im_height = src.height
        im = src.read().transpose(1, 2, 0)
        proj = pyproj.Proj(crs)
        cxs, cys = proj(landmarks[:, 0], landmarks[:, 1])
        lefts, bots = proj(landmarks[:, 2], landmarks[:, 3])
        rights, tops = proj(landmarks[:, 4], landmarks[:, 5])
        cvs, cus = src.index(cxs, cys)
        tlv, tlu = src.index(lefts, tops)
        brv, bru = src.index(rights, bots)
        right_ws = np.array(bru) - np.array(cus)
        bottom_hs = np.array(brv) - np.array(cvs)
        left_ws = np.array(cus) - np.array(tlu)
        top_hs = np.array(cvs) - np.array(tlv)
        ws = np.stack([left_ws, right_ws]).max(axis=0) * 2
        hs = np.stack([top_hs, bottom_hs]).max(axis=0) * 2
        tlv_in = (np.array(tlv) < im_height) * (np.array(tlv) > 0)
        tlu_in = (np.array(tlu) < im_width) * (np.array(tlu) > 0)
        brv_in = (np.array(brv) < im_height) * (np.array(brv) > 0)
        bru_in = (np.array(bru) < im_width) * (np.array(bru) > 0)
        inbounds = tlv_in * tlu_in * brv_in * bru_in
        indexes = np.where(inbounds)
        cvs = np.array(cvs)
        cus = np.array(cus)
        ws_idx = ws[indexes]
        hs_idx = hs[indexes]
        cvs_idx = cvs[indexes]
        cus_idx = cus[indexes]
        classes = indexes[0]
        ws_norm = ws_idx / im_width
        hs_norm = hs_idx / im_height
        cvs_norm = cvs_idx / im_height
        cus_norm = cus_idx / im_width
        out_labels = np.stack([classes, cus_norm, cvs_norm, ws_norm, hs_norm], axis=1)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print('made output directory')
        if not os.path.exists(os.path.join(output_path, 'labels')):
            os.makedirs(os.path.join(output_path, 'labels'))
            print('made labels directory')
        if not os.path.exists(os.path.join(output_path, 'images')):
            os.makedirs(os.path.join(output_path, 'images'))
            print('made images directory')
        file_name = os.path.basename(raster_path).split('.')[0]
        with open(os.path.join(output_path, 'labels', file_name + '.txt'), 'w') as f:
            for label in out_labels:
                f.write(str(int(label[0])) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' + str(label[4]) + '\n')
        im = Image.fromarray(im)
        im.save(os.path.join(output_path, 'images', file_name + '.png'))
        if args.viz_labels:
            if not os.path.exists(os.path.join(output_path, 'viz_labels')):
                os.makedirs(os.path.join(output_path, 'viz_labels'))
                print('made viz directory')
            out_im = visualize_label(os.path.join(output_path, 'labels', file_name + '.txt'), os.path.join(output_path, 'images', file_name + '.png'))
            cv2.imwrite(os.path.join(output_path, 'viz_labels', file_name + '.jpg'), out_im)


def visualize_label(label_path, image_path):
    """
    Visualize the labels on the image.

    Parameters:
    label_path (str): The path to the label file.
    image_path (str): The path to the image file.
    """
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [label.strip().split(' ') for label in labels]
    labels = [[int(label[0]), float(label[1]), float(label[2]), float(label[3]), float(label[4])] for label in labels]
    im = cv2.imread(image_path)
    height, width = im.shape[:2]
    for label in labels:
        left = int((label[1] - label[3]/2) * width)
        top = int((label[2] - label[4]/2) * height)
        right = int((label[1] + label[3]/2) * width)
        bottom = int((label[2] + label[4]/2) * height)
        cv2.rectangle(im, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(im, str(label[0]), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return im

args = parse_args()
landmarks = get_landmarks(args.landmark_path)
raster_paths = get_raster_paths(args.raster_dir_path)
output_path = args.output_dir_path

if __name__ == '__main__':
    process_map(label_raster, raster_paths, max_workers=8, chunksize=1)
