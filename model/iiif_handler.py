import requests
import cv2
import numpy as np
import json
import pprint as pp
import math
from urllib.parse import urlparse, unquote
import os
import uuid


class IIIFHandler:
    def __init__(self, manifest_url, output_dir, img_filename):
        self.tile_info = {'tile_idxs': {}, 'num_tiles_w': 0, 'num_tiles_h': 0}
        # self.url = "https://map-view.nls.uk/iiif/2/12563%2F125635459/12288,8192,4096,3242/512,/0/default.jpg"
        # self.url = "https://map-view.nls.uk/iiif/2/12563%2F125635459/full/max/0/default.jpg"
        self.tile_width = None
        self.tile_height = None
        self.img_width = None
        self.img_height = None
        self.url_prefix = None
        # self.manifest_url = "https://map-view.nls.uk/iiif/2/12563%2F125635459/info.json"
        self.manifest_url = manifest_url
        self.output_dir = output_dir
        self.rotation = 0
        self.tile_size = "full"
        self.quality = "default"
        self.img_format = "jpg"
        self.img_filename = img_filename

    def process_url(self):
        r = requests.get(self.manifest_url)
        # print(r.status_code)
        # print(str(r.headers))

        response_dict = r.json()
        print(json.dumps(response_dict, indent=2))

        self.url_prefix = response_dict['@id']
        # self.img_filename = unquote(urlparse(self.url_prefix).path).split("/")[-1]

        self.img_width = response_dict['width']
        self.img_height = response_dict['height']


        if response_dict['profile'] is not None:
            profile_list = response_dict['profile']
            if type(profile_list) == list and len(profile_list) > 1:
                profile_info = profile_list[1]
                if 'qualities' in profile_info:
                    if 'native' in profile_info['qualities']:
                        self.quality = 'native'
                        print('set to native')

        if response_dict['tiles'] is not None:
            #assert response_dict['tiles'][0]['width'] == response_dict['tiles'][0]['height']
            #tile_size = response_dict['tiles'][0]['width']
            #self.tile_size = str(tile_size) + ','

            tile_info = response_dict['tiles'][0]
            self.tile_width = tile_info['width']
            # hack for sanborn maps
            if 'height' in tile_info:
                self.tile_height = tile_info['height']
            else:
                self.tile_height = tile_info['width']


            assert self.tile_height == self.tile_width

            # hack for david rumsey maps
            try:
                # probe once to decide the url format
                probe_bbox_str = ",".join([str(0), str(0), str(self.tile_width), str(self.tile_height)]) 
                probe_url = self.url_prefix + f"/{probe_bbox_str}/{self.tile_size}/{self.rotation}/{self.quality}.{self.img_format}"
                probe_resp = requests.get(probe_url)
                probe_img = np.asarray(bytearray(probe_resp.content), dtype=np.uint8)
                _,_,_ = prob_img.shape # DO NOT delete this line. This line would cause an error and trigger the execption branch if url format is incorrect
            except:
                
                self.tile_size = str(self.tile_height) + ','


            self._generate_tile_info()
            # pp.pprint(self.tile_info)
            self._download_tiles()
            map_path = self._generate_img()
            return map_path



    # generate a list of unique urls for each tile to download the entire image in pieces
    # https://iiif.io/api/image/2.1/#appendices
    def _generate_tile_info(self):
        row_idx = 0
        col_idx = 0

        max_col_idx = math.ceil(self.img_width / self.tile_width)
        max_row_idx = math.ceil(self.img_height / self.tile_height)

        current_region_x = col_idx * self.tile_width
        current_region_w = self.tile_width
        current_region_y = row_idx * self.tile_height
        current_region_h = self.tile_height

        while col_idx < max_col_idx:
            row_idx = 0 # always start outer loop from new row
            current_region_x = col_idx * self.tile_width
            current_region_w = self.tile_width
            if current_region_x + current_region_w > self.img_width:
                current_region_w = self.img_width - current_region_x

            while row_idx < max_row_idx:
                current_region_y = row_idx * self.tile_height
                current_region_h = self.tile_height

                if current_region_y + current_region_h > self.img_height:
                    current_region_h = self.img_height - current_region_y

                url = self._generate_url(current_region_x, current_region_y, current_region_w, current_region_h)
                self.tile_info['tile_idxs'][(col_idx, row_idx)] = {'url': url}

                row_idx += 1

            col_idx += 1

        url = self._generate_url(current_region_x, current_region_y, current_region_w, current_region_h)
        self.tile_info['tile_idxs'][(col_idx, row_idx)] = {'url': url}

        self.tile_info['num_tiles_w'] = max_col_idx
        self.tile_info['num_tiles_h'] = max_row_idx

    def _download_tiles(self):

        for tile_idx in list(self.tile_info['tile_idxs'].keys()):
            url = self.tile_info['tile_idxs'][tile_idx]['url']

            print(f"downloading for key {str(tile_idx)} - {url}")

            resp = requests.get(url)
            #print(url)
            img = np.asarray(bytearray(resp.content), dtype=np.uint8)

            if img.shape[0] == 0: # empty image
                continue 
        
            try:
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img_height, img_width, img_depth = img.shape
                print(img.shape)

            except:
                print('Error processing tile, stopped', url)
                exit(-1)

            try:
                # Pad width and height to multiples of self.tile_width and self.tile_height
                d_height = self.tile_height - img_height
                d_width = self.tile_width - img_width
                top = 0
                bottom = d_height
                left = 0
                right = d_width

                img = cv2.copyMakeBorder(img.copy(), top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                self.tile_info['tile_idxs'][tile_idx]['img'] = img
            except:
                print('Error making border, continued', url)


    def _generate_img(self):
        num_tiles_w = self.tile_info['num_tiles_w']
        num_tiles_h = self.tile_info['num_tiles_h']

        enlarged_width = int(self.tile_width * num_tiles_w)
        enlarged_height = int(self.tile_height * num_tiles_h)
        print(f"ntw, nth: {num_tiles_h}, {num_tiles_w}")
        print(f"ew, eh: {enlarged_width}, {enlarged_height}")

        # print("BLAGALHAGLAHGA:")
        # print(f"{width}-{num_tiles_w}, {height}-{num_tiles_h}, {enlarged_width}, {enlarged_height}")
        # paste the original map to the enlarged map
        enlarged_map = np.zeros((enlarged_height, enlarged_width, 3)).astype(np.uint8)

        # process tile by tile
        for idx in range(0, num_tiles_w):
            # paste the predicted probabilty maps to the output image
            for jdx in range(0, num_tiles_h):
                if 'img' not in self.tile_info['tile_idxs'][(idx, jdx)]:
                    continue 

                img = self.tile_info['tile_idxs'][(idx, jdx)]['img']
        
                # print(f"img shape for ({idx}, {jdx}) - {img.shape}")
                enlarged_map[jdx * self.tile_width:(jdx + 1) * self.tile_width, idx * self.tile_height:(idx + 1) * self.tile_height, :] = img

        map_path = os.path.join(self.output_dir, self.img_filename)
        cv2.imwrite(map_path, enlarged_map)

        return map_path

    def _generate_url(self, x, y, w, h):

        bbox_str = ",".join([str(x), str(y), str(w), str(h)])
        return_url = self.url_prefix + f"/{bbox_str}/{self.tile_size}/{self.rotation}/{self.quality}.{self.img_format}"
        #print(return_url)
        return return_url

