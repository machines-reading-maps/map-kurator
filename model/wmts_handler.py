import requests
import xmltodict
import json
import shapely.geometry
import math
import cv2
import numpy as np
import os


class WMTSHandler:
    def __init__(self, url, bounds, zoom, output_dir, img_filename):
        self.url = url
        self.tile_info = {}
        self.bounds = json.loads(bounds)
        self.zoom = zoom
        self.output_dir = output_dir
        self.img_filename = img_filename

    def _tile_idxs_in_poly(self, poly: shapely.geometry.Polygon):
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        (min_x, max_y), (max_x, min_y) = self._latlon2tile(min_lat, min_lon), self._latlon2tile(max_lat, max_lon)

        tile_idxs = []

        for x in range(int(min_x), int(max_x) + 1):
            for y in range(int(min_y), int(max_y) + 1):
                nw_pt = self._tile2latlon(x, y)[::-1]  # poly is defined in geojson form
                ne_pt = self._tile2latlon(x + 1, y)[::-1]  # poly is defined in geojson form
                sw_pt = self._tile2latlon(x, y + 1)[::-1]  # poly is defined in geojson form
                se_pt = self._tile2latlon(x + 1, y + 1)[::-1]  # poly is defined in geojson form

                bbox = shapely.geometry.Polygon([nw_pt, ne_pt, sw_pt, se_pt])

                # print(f"{x}-{y}; {nw_pt} {ne_pt} {sw_pt} {se_pt}")
                # if any(map(lambda pt: shapely.geometry.Point(pt).within(poly), (nw_pt, ne_pt, sw_pt, se_pt))):
                if poly.intersects(bbox):
                    tile_idxs.append((x, y))

        return tile_idxs, int(max_x + 1) - int(min_x), int(max_y + 1) - int(min_y), int(min_x), int(min_y)

    def _generate_tile_info(self, tile_idxs, min_x, min_y, url_template):
        zoom_level = str(self.zoom)
        tile_info = {
            'zoom_level': zoom_level,
            'tile_idxs': {}
        }

        for (x, y) in tile_idxs:
            # tile_col = str(x)
            # tile_row = str(y)

            url = url_template.replace('{TileMatrix}', zoom_level).replace('{TileCol}', str(x)).replace('{TileRow}',
                                                                                                        str(y))
            tile_info['tile_idxs'][(x - min_x, y - min_y)] = {'url': url}

        return tile_info

    def process_wmts(self):
        # print(args)
        # zoom_level = 18 # ~45min to download and predict; similar results to zoom=16; stitched png ~100Mb
        # zoom_level = 16 # ~2 min to download to predict; decent results; stitched png ~7Mb
        # zoom_level = 14 # too small for the model to detect text

        r = requests.get(self.url)
        # print(r.status_code)
        # print(str(r.headers))
        # print(json.dumps(xmltodict.parse(r.content)))
        response_dict = xmltodict.parse(r.content)
        wmts_capabilities = response_dict['Capabilities']
        # print(list(wmts_capabilities.keys()))
        url_template = wmts_capabilities['Contents']['Layer']['ResourceURL']['@template']

        poly = shapely.geometry.shape(self.bounds['geometry'])

        tile_idxs, num_tiles_w, num_tiles_h, min_x, min_y = self._tile_idxs_in_poly(poly)

        # print(f"num_tiles: {len(tile_idxs)}")
        tile_info = self._generate_tile_info(tile_idxs, min_x, min_y, url_template)
        tile_info['num_tiles_w'] = num_tiles_w
        tile_info['num_tiles_h'] = num_tiles_h
        tile_info['min_x'] = min_x 
        tile_info['min_y'] = min_y

        tile_info = self._download_tiles(tile_info)

        map_path = self._generate_img(tile_info)

        # update self.tile_info
        self.tile_info = tile_info

        return map_path

    def _download_tiles(self, tile_info):

        for tile_idx in list(tile_info['tile_idxs'].keys()):
            url = tile_info['tile_idxs'][tile_idx]['url']

            print(f"downloading for key {str(tile_idx)} - {url}")

            resp = requests.get(url)
            img = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            tile_info['tile_idxs'][tile_idx]['img'] = img

        # return the images
        return tile_info

    def _generate_img(self, tile_info):
        num_tiles_w = tile_info['num_tiles_w']
        num_tiles_h = tile_info['num_tiles_h']

        shift_size = 256

        enlarged_width = int(shift_size * num_tiles_w)
        enlarged_height = int(shift_size * num_tiles_h)

        # paste the original map to the enlarged map
        enlarged_map = np.zeros((enlarged_height, enlarged_width, 3)).astype(np.uint8)

        # process tile by tile
        for idx in range(0, max(1,num_tiles_w)):
            # paste the predicted probabilty maps to the output image
            for jdx in range(0, max(1,num_tiles_h)):
                img = tile_info['tile_idxs'][(idx, jdx)]['img']
                enlarged_map[jdx * shift_size:(jdx + 1) * shift_size, idx * shift_size:(idx + 1) * shift_size, :] = img

        map_path = os.path.join(self.output_dir, self.img_filename)

        cv2.imwrite(map_path, enlarged_map)
        return map_path

    def _stitch_tiles(self):
        # needs input path with (cached) image tiles
        # needs output path
        return True

    # from OSM Slippy Tile definitions & https://github.com/Caged/tile-stitch
    def _latlon2tile(self, lat, lon):
        lat_radians = lat * math.pi / 180.0
        n = 1 << self.zoom
        return (
            n * ((lon + 180.0) / 360.0),
            n * (1 - (math.log(math.tan(lat_radians) + 1 / math.cos(lat_radians)) / math.pi)) / 2.0
        )

    # from OSM Slippy Tile definitions & https://github.com/Caged/tile-stitch
    def _tile2latlon(self, x, y):
        n = 1 << self.zoom
        lat_radians = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
        lat = lat_radians * 180 / math.pi
        lon = 360 * x / n - 180.0
        return (lat, lon)

    def _tile2latlon_list(self, x_list, y_list):
        n = 1 << self.zoom
        x_list, y_list = np.array(x_list), np.array(y_list)
        lat_radians_list = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y_list / n)))
        lat_list = lat_radians_list * 180 / math.pi
        lon_list = 360 * x_list / n - 180.0
        return (lat_list, lon_list)
