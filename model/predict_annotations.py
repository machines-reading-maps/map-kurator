import argparse
import os
import xml.etree.ElementTree as ET
from wmts_handler import WMTSHandler






parser = argparse.ArgumentParser()
# parser.add_argument("input_type", choices=["wmts", "iiif", "tiff", "jpeg", "png"])
subparsers = parser.add_subparsers(dest='subcommand')

arg_parser_wmts = subparsers.add_parser('wmts')
# arg_parser_wmts.add_argument('url')
# arg_parser_wmts.add_argument('boundary')
# arg_parser_wmts.add_argument('zoom-level', type=int)

arg_parser_iiif = subparsers.add_parser('iiif')
arg_parser_iiif.add_argument('--url')

arg_parser_raw_input = subparsers.add_parser('raw-input')
arg_parser_raw_input.add_argument('input-path')

parser.add_argument('output')
args = parser.parse_args()

def process_iiif(args):
    print("This is IIIF mock")
    print(args)


def process_raw_input(args):
    print("This is raw input mock")
    print(args)


if args.subcommand == 'wmts':
    # process_wmts(args)
    print("hi")
    wmts_handler = WMTSHandler()
    wmts_handler.process_wmts(2)


if args.subcommand == 'iiif':
    # process_wmts(args)
    print("hi")

if args.subcommand == 'raw-input':
    process_raw_input(args)



