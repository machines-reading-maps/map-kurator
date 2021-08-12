import cv2
import numpy as np
import rasterio


class ImageHandler:
    def __init__(self):
        self.img = None
        print("ImageHandler")

    def process_img(self, args):
        print(f"ImageHandler: {args}")

