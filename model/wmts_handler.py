import requests
import xmltodict
import json
import shapely.geometry
import math
import cv2
import numpy as np


class WMTSHandler:
    def __init__(self):
        self.url = 1
        self.tile_info = {}
        self.bounds = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-1.1248, 53.9711],
                    [-1.0592, 53.9711],
                    [-1.0592, 53.9569],
                    [-1.1248, 53.9569],
                    [-1.1248, 53.9711]
                ]]
            }
        }

    def _tile_idxs_in_poly(self, poly: shapely.geometry.Polygon, zoom: int):
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        (min_x, max_y), (max_x, min_y) = self._latlon2tile(min_lat, min_lon, zoom), self._latlon2tile(max_lat, max_lon, zoom)

        tile_idxs = []

        for x in range(int(min_x), int(max_x) + 1):
            for y in range(int(min_y), int(max_y) + 1):
                nw_pt = self._tile2latlon(x, y, zoom)[::-1]  # poly is defined in geojson form
                ne_pt = self._tile2latlon(x + 1, y, zoom)[::-1]  # poly is defined in geojson form
                sw_pt = self._tile2latlon(x, y + 1, zoom)[::-1]  # poly is defined in geojson form
                se_pt = self._tile2latlon(x + 1, y + 1, zoom)[::-1]  # poly is defined in geojson form

                bbox = shapely.geometry.Polygon([nw_pt, ne_pt, sw_pt, se_pt])

                # print(f"{x}-{y}; {nw_pt} {ne_pt} {sw_pt} {se_pt}")
                # if any(map(lambda pt: shapely.geometry.Point(pt).within(poly), (nw_pt, ne_pt, sw_pt, se_pt))):
                if poly.intersects(bbox):
                    tile_idxs.append((x, y))

        return tile_idxs, int(max_x + 1) - int(min_x), int(max_y + 1) - int(min_y), int(min_x), int(min_y)

    def _download_tiles(self):
        return True

    def _generate_tile_info(self, tile_idxs, min_x, min_y, zoom_level, url_template):
        zoom_level = str(zoom_level)
        tile_info = {
            'zoom_level': zoom_level,
            'tile_idxs': {}
        }

        print(tile_idxs)

        for (x, y) in tile_idxs:
            # tile_col = str(x)
            # tile_row = str(y)

            url = url_template.replace('{TileMatrix}', zoom_level).replace('{TileCol}', str(x)).replace('{TileRow}',
                                                                                                        str(y))
            tile_info['tile_idxs'][(x - min_x, y - min_y)] = {
                'url': url
            }

        return tile_info

    def process_wmts(self, args):
        print("This is WMTS mock")
        # print(args)
        zoom_level = 14
        r = requests.get(
            "https://wmts.maptiler.com/aHR0cDovL3dtdHMubWFwdGlsZXIuY29tL2FIUjBjSE02THk5dFlYQnpaWEpwWlhNdGRHbHNaWE5sZEhNdWN6TXVZVzFoZW05dVlYZHpMbU52YlM4eU5WOXBibU5vTDNsdmNtdHphR2x5WlM5dFpYUmhaR0YwWVM1cWMyOXUvanNvbg/wmts")
        # print(r.status_code)
        # print(str(r.headers))
        # print(json.dumps(xmltodict.parse(r.content)))
        response_dict = xmltodict.parse(r.content)
        wmts_capabilities = response_dict['Capabilities']
        # print(list(wmts_capabilities.keys()))
        url_template = wmts_capabilities['Contents']['Layer']['ResourceURL']['@template']
        # print(url_template)

        poly = shapely.geometry.shape(self.bounds['geometry'])

        tile_idxs, num_tiles_w, num_tiles_h, min_x, min_y = self._tile_idxs_in_poly(poly, zoom_level)

        # print(f"num_tiles: {len(tile_idxs)}")
        tile_info = self._generate_tile_info(tile_idxs, min_x, min_y, zoom_level, url_template)
        tile_info['num_tiles_w'] = num_tiles_w
        tile_info['num_tiles_h'] = num_tiles_h

        tile_info = self._download_tiles(tile_info)

        self._generate_img(tile_info)

        print("Running predictions...")
        exec(open("model/save_localheight_original_txt_fastzk.py").read())

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
                img = tile_info['tile_idxs'][(idx, jdx)]['img']
                print(f"img shape for ({idx}, {jdx}) - {img.shape}")
                enlarged_map[jdx * shift_size:(jdx + 1) * shift_size, idx * shift_size:(idx + 1) * shift_size, :] = img

        cv2.imwrite('data/stitched.png', enlarged_map)

    def _stitch_tiles(self):
        # needs input path with (cached) image tiles
        # needs output path
        return True

    # from OSM Slippy Tile definitions & https://github.com/Caged/tile-stitch
    def _latlon2tile(self, lat, lon, zoom):
        lat_radians = lat * math.pi / 180.0
        n = 1 << zoom
        return (
            n * ((lon + 180.0) / 360.0),
            n * (1 - (math.log(math.tan(lat_radians) + 1 / math.cos(lat_radians)) / math.pi)) / 2.0
        )

    # from OSM Slippy Tile definitions & https://github.com/Caged/tile-stitch
    def _tile2latlon(self, x, y, zoom):
        n = 1 << zoom
        lat_radians = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
        lat = lat_radians * 180 / math.pi
        lon = 360 * x / n - 180.0
        return (lat, lon)