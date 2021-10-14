import argparse
import os
import uuid

from wmts_handler import WMTSHandler
from image_handler import ImageHandler
from iiif_handler import IIIFHandler
from mymodel import model_U_VGG_Centerline_Localheight

import cv2
import numpy as np
import json

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import sys
import tensorflow as tf

import time

print(tf.__file__)
print(tf.__version__)

# basically copy-pasted from the original implementation in save_localheight_original_txt_fastzk.py
def run_model(map_id, map_path, output_dir):
    saved_weights = './data/l_weights/finetune_map_model_map_w1e50_bsize8_w1_spe200_ep50.hdf5'
    model = model_U_VGG_Centerline_Localheight()
    model.load_weights(saved_weights)

    map_img = cv2.imread(map_path)
    shift_size = 512

    base_name = os.path.basename(map_path)

    width = map_img.shape[1]  # dimension2
    height = map_img.shape[0]  # dimension1

    in_map_img = map_img / 255.

    # pad the image to the size divisible by shift-size
    num_tiles_w = int(np.ceil(1. * width / shift_size))
    num_tiles_h = int(np.ceil(1. * height / shift_size))
    enlarged_width = int(shift_size * num_tiles_w)
    enlarged_height = int(shift_size * num_tiles_h)
    # print(f"{width}-{num_tiles_w}, {height}-{num_tiles_h}, {enlarged_width}, {enlarged_height}")
    # paste the original map to the enlarged map
    enlarged_map = np.zeros((enlarged_height, enlarged_width, 3)).astype(np.float32)
    enlarged_map[0:height, 0:width, :] = in_map_img

    # define the output probability maps
    localheight_map_o = np.zeros((enlarged_height, enlarged_width, 1), np.float32)
    center_map_o = np.zeros((enlarged_height, enlarged_width, 2), np.float32)
    prob_map_o = np.zeros((enlarged_height, enlarged_width, 3), np.float32)

    # process tile by tile
    for idx in range(0, num_tiles_h):
        # pack several tiles in a batch and feed the batch to the model
        test_batch = []
        for jdx in range(0, num_tiles_w):
            img_clip = enlarged_map[idx * shift_size:(idx + 1) * shift_size, jdx * shift_size:(jdx + 1) * shift_size, :]
            test_batch.append(img_clip)
        test_batch = np.array(test_batch).astype(np.float32)

        # use the pretrained model to predict
        batch_out = model.predict(test_batch)

        # get predictions
        prob_map_batch = batch_out[0]
        center_map_batch = batch_out[1]
        localheight_map_batch = batch_out[2]

        # paste the predicted probabilty maps to the output image
        for jdx in range(0, num_tiles_w):
            localheight_map_o[idx * shift_size:(idx + 1) * shift_size, jdx * shift_size:(jdx + 1) * shift_size, :] = \
            localheight_map_batch[jdx]
            center_map_o[idx * shift_size:(idx + 1) * shift_size, jdx * shift_size:(jdx + 1) * shift_size, :] = \
            center_map_batch[jdx]
            prob_map_o[idx * shift_size:(idx + 1) * shift_size, jdx * shift_size:(jdx + 1) * shift_size, :] = \
            prob_map_batch[jdx]

    # convert from 0-1? to 0-255 range
    prob_map_o = (prob_map_o * 255).astype(np.uint8)
    center_map_o = (center_map_o[:, :, 1] * 255).astype(np.uint8)
    # localheight_map = (localheight_map_o * 255).astype(np.uint8)

    prob_map_o = prob_map_o[0:height, 0:width, :]
    center_map_o = center_map_o[0:height, 0:width]
    localheight_map_o = localheight_map_o[0:height, 0:width, :]

    num_c, connected_map = cv2.connectedComponents(center_map_o)
    print('num_c:', num_c)

    poly_list = []
    # process component by component
    for cur_cc_idx in range(1, num_c):  # index_0 is the background

        if cur_cc_idx % 100 == 0:
            print('processed', str(cur_cc_idx))

        centerline_indices = np.where(connected_map == cur_cc_idx)

        centerPoints = []
        for i, j in zip(centerline_indices[0], centerline_indices[1]):
            if localheight_map_o[i, j, 0] > 0:
                centerPoints.append([i, j])

        if len(centerPoints) == 0:
            continue

        mini, minj = np.min(centerPoints, axis=0)
        maxi, maxj = np.max(centerPoints, axis=0)

        localheight_result_o = np.zeros((maxi - mini + 100, maxj - minj + 100, 3), np.uint8)

        for i, j in centerPoints:
            cv2.circle(localheight_result_o, (j - minj + 50, i - mini + 50), int(localheight_map_o[i][j] * 0.5),
                       (0, 0, 255), -1)

        img_gray = cv2.cvtColor(localheight_result_o, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        new_context = ''

        if len(contours) == 0:
            continue

        for i in range(0, len(contours[0])):
            if i < len(contours[0]) - 1:
                new_context = new_context + str(contours[0][i][0][0].item() + minj - 50) + ',' + str(
                    contours[0][i][0][1].item() + mini - 50) + ','
            else:
                new_context = new_context + str(contours[0][i][0][0].item() + minj - 50) + ',' + str(
                    contours[0][i][0][1].item() + mini - 50)

        # new_context = new_context + '\n'
        poly_str = new_context.split(',')
        poly = []
        for i in range(0, len(poly_str)):
            if i % 2 == 0:
                poly.append([int(poly_str[i]), int(poly_str[i + 1])])

        poly_list.append(poly)

        # cv2.imwrite(output_path + 'prob_' + base_name[0:len(base_name) - 4] + '.jpg', prob_map_o)
        # cv2.imwrite(output_path + 'cent_' + base_name[0:len(base_name) - 4] + '.jpg', center_map_o)
        # cv2.imwrite(output_path + 'localheight_map_' + base_name[0:len(base_name) - 4] + '.jpg', localheight_map_o)


    for i in range(0,len(poly_list)):
        poly_points = np.array([poly_list[i]], dtype=np.int32)
        cv2.polylines(map_img, poly_points, True, (0, 0, 255), 3)

    predictions_file = os.path.join(output_dir, map_id + '_predictions.jpg')
    cv2.imwrite(predictions_file, map_img)


    return poly_list

def write_annotation(map_id, output_dir, poly_list, handler = None):


    if handler is not None: 
        # perform this operation for WMTS tiles only
        # based on the tile info, convert from image coordinate system to EPSG：4326
        # assumes that the tilesize = 256x256

        tile_info = handler.tile_info

        min_tile_x = tile_info['min_x']
        min_tile_y = tile_info['min_y']

        latlon_poly_list = []
        for polygon in poly_list:
            # process each polygon 
            poly_x_list , poly_y_list = np.array(polygon)[:,0], np.array(polygon)[:,1] 

            # get corresponding tile index in the current map, i.e. tile shift range from min_tile_x ,min_tile_y
            temp_tile_x_list, temp_tile_y_list = np.floor(poly_x_list/ 256.),  np.floor(poly_y_list/256.)

            # compute the starting tile idx that the polygon point lies in
            tile_x_list, tile_y_list = min_tile_x + temp_tile_x_list , min_tile_y + temp_tile_y_list

            # get polygon point pixel location in its current tile
            remainder_x_list, remainder_y_list = poly_x_list/256. - temp_tile_x_list , poly_y_list/256. - temp_tile_y_list

            # final position in EPSG:3857? 
            tile_x_list, tile_y_list = tile_x_list + remainder_x_list, tile_y_list + remainder_y_list  

            # convert to EPSG:4326
            lat_list, lon_list = handler._tile2latlon_list(tile_x_list, tile_y_list)

            latlon_poly = [[x,y] for x,y in zip(lat_list, lon_list)]
            latlon_poly_list.append(latlon_poly)

        poly_list = latlon_poly_list
        # reassign latlon_poly_list to poly_list for consistency


    # Generate web annotations: https://www.w3.org/TR/annotation-model/
    annotations = []
    for polygon in poly_list:
        svg_polygon_coords = ' '.join([f"{x},{y}" for x, y in polygon])
        annotation = {
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "id": "",
            "body": [{
                "type": "TextualBody",
                "purpose": "tagging",
                "value": "null"
            }],
            "target": {
                "selector": [{
                    "type": "SvgSelector",
                    "value": f"<svg><polygon points='{svg_polygon_coords}'></polygon></svg>"
                }]
            }
        }
        annotations.append(annotation)

    annotation_file = os.path.join(output_dir, map_id + '_annotations.json')
    with open(annotation_file, 'w') as f:
        f.write(json.dumps(annotations, indent=2))

    return annotation_file
    # print(f"{polyList}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    arg_parser_common = argparse.ArgumentParser(add_help=False)
    arg_parser_common.add_argument('--dst', required=True, type=str, help='path to output annotations file')
    arg_parser_common.add_argument('--filename', required=False, type=str, help='output filename prefix')
    arg_parser_common.add_argument('--coord', default = 'img_coord', required=False, type=str, choices = ['img_coord' ,'epsg4326'], help='return annotation in image coord or EPSG:4326')

    # parser.add_argument("input_type", choices=["wmts", "iiif", "tiff", "jpeg", "png"])
    subparsers = parser.add_subparsers(dest='subcommand')

    arg_parser_wmts = subparsers.add_parser('wmts', parents=[arg_parser_common],
                                            help='generate annotations for wmts input type')
    arg_parser_wmts.add_argument('--url', required=True, type=str, help='getCapabilities url')
    arg_parser_wmts.add_argument('--boundary', required=True, type=str, help='desired region boundary in GeoJSON')
    arg_parser_wmts.add_argument('--zoom', default=14, type=int, help='desired zoom level')

    arg_parser_iiif = subparsers.add_parser('iiif', parents=[arg_parser_common],
                                            help='generate annotations for iiif input type')
    arg_parser_iiif.add_argument('--url', required=True, type=str, help='IIIF manifest url')

    arg_parser_raw_input = subparsers.add_parser('file', parents=[arg_parser_common])
    arg_parser_raw_input.add_argument('--src', required=True, type=str, help='path to input image')

    args = parser.parse_args()

    map_path = None
    output_dir = args.dst

    if args.filename is not None:
        img_id = args.filename
    else:
        img_id = str(uuid.uuid4())

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    if args.coord == 'epsg4326':
        assert args.subcommand == 'wmts'


    if args.subcommand == 'wmts':
        '''
time docker run -it -v $(pwd)/data/:/map-kurator/data -v $(pwd)/model:/map-kurator/model --rm --runtime=nvidia --gpus all  --workdir=/map-kurator map-kurator python model/predict_annotations.py wmts --url='https://wmts.maptiler.com/aHR0cDovL3dtdHMubWFwdGlsZXIuY29tL2FIUjBjSE02THk5dFlYQnpaWEpwWlhNdGRHbHNaWE5sZEhNdWN6TXVZVzFoZW05dVlYZHpMbU52YlM4eU5WOXBibU5vTDNsdmNtdHphR2x5WlM5dFpYUmhaR0YwWVM1cWMyOXUvanNvbg/wmts' --boundary='{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[-1.1248,53.9711],[-1.0592,53.9711],[-1.0592,53.9569],[-1.1248,53.9569],[-1.1248,53.9711]]]}}' --zoom=16 --dst=data/test_imgs/sample_output/
        '''

        wmts_handler = WMTSHandler(url=args.url, bounds=args.boundary, zoom=args.zoom, output_dir=output_dir, img_filename=img_id + '_stitched.jpg')
        map_path = wmts_handler.process_wmts()

        poly_list = run_model(img_id, map_path, output_dir)
        if args.coord == 'img_coord':
            annotation_file = write_annotation(img_id, output_dir, poly_list)
        else:
            annotation_file = write_annotation(img_id, output_dir, poly_list, handler = wmts_handler)

    if args.subcommand == 'iiif':
        '''
time docker run -it -v $(pwd)/data/:/map-kurator/data -v $(pwd)/model:/map-kurator/model --rm --runtime=nvidia --gpus all  --workdir=/map-kurator map-kurator python model/predict_annotations.py iiif --url='https://map-view.nls.uk/iiif/2/12563%2F125635459/info.json' --dst=data/test_imgs/sample_output/
        '''
        start_download = time.time()
        iiif_handler = IIIFHandler(args.url, output_dir, img_filename=img_id + '_stitched.jpg')
        map_path = iiif_handler.process_url()

        end_download = time.time()

        poly_list = run_model(img_id, map_path, output_dir)
        annotation_file = write_annotation(img_id, output_dir, poly_list)

        end_detection = time.time()

        print('download time: ', end_download - start_download)
        print('detection time: ', end_detection - end_download)
        

    if args.subcommand == 'file':
        '''
time docker run -it -v $(pwd)/data/:/map-kurator/data -v $(pwd)/model:/map-kurator/model --rm --runtime=nvidia --gpus all  --workdir=/map-kurator map-kurator python model/predict_annotations.py file --src=data/test_imgs/sample_input/101201496_h10w3.jpg --dst=data/test_imgs/sample_output/
        '''
        map_path = args.src

        poly_list = run_model(img_id, map_path, output_dir)
        annotation_file = write_annotation(img_id, output_dir, poly_list)


    

    print("done")
    print(annotation_file)
