import os
import sys
import argparse
import ipdb

import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow_text


###############################################################################

parser = argparse.ArgumentParser('USEm: calculate embeddings')
parser.add_argument(
    '--input_text', type=str, required=True)
parser.add_argument(
    '--output_embeding', type=str, required=True)
args = parser.parse_args()

print('USEm: calculate embeddings')


def USEm(txt_list, usem_embeder):
    output = []
    txt_df = pd.Series(txt_list)
    batch_size = 1024
    for k,g in txt_df.groupby(np.arange(len(txt_df))//batch_size):
        output.append(usem_embeder(g.tolist()).numpy())
    output = np.concatenate(output, axis=0).flatten()
    output.resize(output.shape[0] // 512, 512)
    return output

def main():
    print('\nProcessing:')
    usem_embeder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    txt_list = []
    with open(args.input_text,'r') as f:
        for l in f:
            txt_list.append(l)
                
    usem_embed = USEm(txt_list, usem_embeder)
    usem_embed.tofile(args.output_embeding)  
    
if __name__ == "__main__":
    main()