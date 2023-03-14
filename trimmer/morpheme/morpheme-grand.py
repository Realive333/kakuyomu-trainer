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

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def analysis(n, v, a, path, work):
    noun_list = []
    verb_list = []
    adjv_list = []
    surface_list =[]
    
    cnt = work['content']
    sep = re.split("。|」|？|！", cnt)
    
    mecab = MeCab.Tagger("")
    for sent in sep:
        node = mecab.parseToNode(sent)
        surface = []
        while node:
            surface.append(node.surface)
            term = node.feature.split(",")[0]
            if term == "名詞":
                noun_list.append(node.surface)
            elif term == "動詞":
                verb_list.append(node.surface)
            elif term == "形容詞" or term == "副詞":
                adjv_list.append(node.surface)
            else:
                pass
            node = node.next
        surface_list.append(' '.join(surface))
    
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/{work["id"]}', 'w', encoding='utf-8') as f:
        for line in surface_list:
            f.write(f'{line}\n')
            
    for noun in noun_list:
        n[noun] += 1
    for verb in verb_list:
        v[verb] += 1
    for adjv in adjv_list:
        a[adjv] += 1
    
    return n, v, a

def main(args):
    DATA_DIR = f'./kakuyomu-data/'
    SAVE_DIR = f'./morpheme/grand/'
    
    works = workFactory.read_cleaned_works(f"{DATA_DIR}", args.size)
    
    batch_count = 0
    for work in batch(works, 3000):
        print(f'\tbatch: {batch_count}')
        noun = defaultdict(int)
        verb = defaultdict(int)
        adjv = defaultdict(int)
        for w in tqdm(work):
            noun, verb, adjv = analysis(noun, verb, adjv, f'{SAVE_DIR}/{batch_count}', w)
        noun = OrderedDict(sorted(noun.items(), key=lambda x:x[1], reverse=True))
        verb = OrderedDict(sorted(verb.items(), key=lambda x:x[1], reverse=True))
        adjv = OrderedDict(sorted(adjv.items(), key=lambda x:x[1], reverse=True))

        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(f'{SAVE_DIR}/{batch_count}/noun.tsv', 'w') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['word', 'count'])
            for key, val in noun.items():
                w.writerow([key, val])

        with open(f'{SAVE_DIR}/{batch_count}/verb.tsv', 'w') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['word', 'count'])
            for key, val in verb.items():
                w.writerow([key, val])

        with open(f'{SAVE_DIR}/{batch_count}/adjv.tsv', 'w') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['word', 'count'])
            for key, val in adjv.items():
                w.writerow([key, val])
        batch_count += 1

if __name__ == "__main__":
    parser = ArgumentParser(description= "Grand Morphological Analysis")
    parser.add_argument('--size', type=int, default='-1')
    args = parser.parse_args()
    
    print(f'Grand Morphological Analysis\n\tSize:{args.size}')
    main(args)