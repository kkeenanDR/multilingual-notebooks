#!/opt/miniconda3/envs/MultiLingualEmbeddings/bin/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#


import os
import sys
import argparse
import ipdb
from tqdm import tqdm
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow_text


###############################################################################

parser = argparse.ArgumentParser('LASER: calculate embeddings for MLDoc')
parser.add_argument(
    '--mldoc', type=str, default='MLDoc',
    help='Directory of the MLDoc corpus')
parser.add_argument(
    '--data_dir', type=str, default='embed',
    help='Base directory for created files')

# options for encoder
parser.add_argument(
    '--encoder', type=str, required=False,
    help='Encoder to be used')
parser.add_argument(
    '--bpe_codes', type=str, required=False,
    help='Directory of the tokenized data')
parser.add_argument(
    '--lang', '-L', nargs='+', default=None,
    help="List of languages to test on")
parser.add_argument(
    '--buffer-size', type=int, default=10000,
    help='Buffer size (sentences)')
parser.add_argument(
    '--max-tokens', type=int, default=12000,
    help='Maximum number of tokens to process in a batch')
parser.add_argument(
    '--max-sentences', type=int, default=None,
    help='Maximum number of sentences to process in a batch')
parser.add_argument(
    '--cpu', action='store_true',
    help='Use CPU instead of GPU')
parser.add_argument(
    '--verbose', action='store_true',
    help='Detailed output')
args = parser.parse_args()

print('LASER: calculate embeddings for MLDoc')

if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)

def USEm(txt_list, usem_embedder):
    output = []
    txt_df = pd.Series(txt_list)
    batch_size = 12
    for k,g in tqdm(txt_df.groupby(np.arange(len(txt_df))//batch_size)):
        output.append(usem_embedder(g.tolist()).numpy())
    output = np.concatenate(output, axis=0).flatten()
    print(output.shape)
    print(output.min())
    print(output.max())
    print('\n')
    return output

def main():
    print('\nProcessing:')
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    for part in ('train', 'dev', 'test'):
        #for lang in "en" if part == 'train' else args.lang:
        for lang in args.lang:
            if (lang != 'en') and (part in ['train','dev']):
                continue
            cfname = os.path.join(args.data_dir, part)
            if os.path.exists(cfname + '.usem.enc.' + lang):
                print(cfname + '.usem.enc.' + lang + ' Exists')
                continue

            txt = []
            with open(cfname + '.txt.' + lang) as f:
                for l in f:
                    txt.append(l)
            usem_np = USEm(txt, embed)
            print(usem_np.dtype, usem_np.shape)

            usem_np.tofile(cfname + '.usem.enc.' + lang)
            #np.save(cfname + '.usem.enc.' + lang, usem_np)

if __name__ == "__main__":
    main()