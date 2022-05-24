import os
import sys
import argparse
import ipdb

import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow_text


###############################################################################

parser = argparse.ArgumentParser('LASER: calculate embeddings')
parser.add_argument(
    '--lang', type=str, required=True)
parser.add_argument(
    '--input_text', type=str, required=True)
parser.add_argument(
    '--output_embeding', type=str, required=True)
parser.add_argument(
    '--LASER_path', type=str, required=True)
args = parser.parse_args()

print('LASER: calculate embeddings')


def main():
    print('\nGenerate LASER Embedding :')                
    os.system(
        '{LASER_path}/tasks/embed/embed.sh {input_text} {lang} {output}'.format(
            LASER_path=args.LASER_path, 
            input_text=args.input_text, 
            lang=args.lang, 
            output=args.output_embeding
        )
    )
    
if __name__ == "__main__":
    main()