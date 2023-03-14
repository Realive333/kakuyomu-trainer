import os
import re
import sys
import csv
import time
import json
import itertools

import MeCab
import workFactory

from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from collections import OrderedDict


def analysis(work):
    surface = []
    cnt = work['content']
    sep = re.split("。|「|」|？|！", cnt)
    mecab = MeCab.Tagger("-Owakati")
    for sent in sep:
        try:
            phrase = mecab.parse(sent).replace('\n', '')
        except AttributeError:
            print(sent)
        surface.append(phrase)
    surface = [x for x in surface if x]
    return '\n'.join(surface)

def getMorphemeByTSV(path, dataType):
    workList = []
    works = workFactory.load_tsv_dataset(f'{path}/{dataType}.tsv')
    for work in tqdm(works):    ##
        morp = analysis(work)
        if work['label'] == 0:
            label = '01'
        elif work['label'] == 1:
            label = '10'
        workList.append({'label': label, 'content': morp})
    return workList

def saveMorphemeAsTSV(workDir, saveDir, dataType):
    works = getMorphemeByTSV(f'{workDir}', dataType)
    with open(f'{saveDir}/{dataType}.tsv','w' ,encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        for work in works:
            w.writerow([work['label'], work['content']])
        

def main(args):
    csv.field_size_limit(sys.maxsize)
    WORK_DIR = f'./tsv/full/{args.target}'
    SAVE_DIR = f'./tsv/morpheme/{args.target}'
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    saveMorphemeAsTSV(WORK_DIR, SAVE_DIR, 'train')
    saveMorphemeAsTSV(WORK_DIR, SAVE_DIR, 'test')
    saveMorphemeAsTSV(WORK_DIR, SAVE_DIR, 'dev')

if __name__ == "__main__":
    parser = ArgumentParser(description = "Morphological TSV Analysis")
    parser.add_argument('--target', type=str, default='1')
    args = parser.parse_args()
    
    main(args)