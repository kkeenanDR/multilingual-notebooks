#!/usr/bin/python

import os
import sys
import argparse
import ipdb
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from fairseq.models.roberta import XLMRModel
from torchnlp.encoders.text import stack_and_pad_tensors
import torch


###############################################################################

parser = argparse.ArgumentParser('XLMRR:')
parser.add_argument(
    '--data_dir', type=str, default='embed',
    help='Base directory for created files')

# options for encoder
parser.add_argument(
    '--encoder', type=str, required=True,
    help='Encoder to be used')
parser.add_argument(
    '--bpe_codes', type=str, required=True,
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


if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)

def batch_encoder(text_list, tokenizer):
    batch = []
    for sequence in text_list:
        batch.append(tokenizer.encode(sequence))
    return stack_and_pad_tensors(batch, tokenizer.task.source_dictionary.__dict__["indices"]["<pad>"])


def XLMR(txt_list, xlmr):
    output = []
    txt_df = pd.Series(txt_list)
    batch_size = 16
    for k,g in tqdm(txt_df.groupby(np.arange(len(txt_df))//batch_size)):
        g = g.str.slice(0,128)
        batch_tokens, batch_lengths = batch_encoder(g.tolist(), xlmr)
        batch_features = xlmr.extract_features(batch_tokens)
        for idx, txt_length in enumerate(batch_lengths):
            output.append(batch_features[idx][:txt_length].mean(axis=0).cpu().detach().numpy())
        torch.cuda.empty_cache()
    output = np.vstack(output)
    return output

def main():
    print('\nProcessing:')
    xlmr = XLMRModel.from_pretrained("../../generate_embeddings/xlmr.large", checkpoint_file="model.pt")
    xlmr.eval() # disable dropout (or leave in train mode to finetune)
    if torch.cuda.is_available():
        xlmr.cuda()

    for part in ('train', 'dev', 'test'):
        #for lang in "en" if part == 'train' else args.lang:
        for lang in args.lang:
            if (lang != 'en') and (part in ['train','dev']):
                continue
            cfname = os.path.join(args.data_dir, part)
            if os.path.exists(cfname + '.xlmr.enc.' + lang):
                print(cfname + '.xlmr.enc.' + lang + ' Exists')
                continue

            txt = []
            with open(cfname + '.txt.' + lang) as f:
                for l in f:
                    txt.append(l)
            xlmr_np = XLMR(txt, xlmr)
            print(xlmr_np.dtype, xlmr_np.shape)

            xlmr_np.tofile(cfname + '.xlmr.enc.' + lang)
            #np.save(cfname + '.usem.enc.' + lang, usem_np)

if __name__ == "__main__":
    main()
