import os
import re
import sys
import csv
import time
import json
import itertools

import MeCab
import pandas as pd
import work

from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from multiprocessing import Pool
from collections import defaultdict
from collections import OrderedDict

def findPositive(works):
    pos_works = [work for work in works if not work['label']==0]
    for w in pos_works:
        if w['label'] == 0:
            raise Exception('All works should be positive example, has', work)
    return pos_works

def analyze(n, v, a, text):
    noun_list = []
    verb_list = []
    adjv_list = []
    mecab = MeCab.Tagger("")
    
    node = mecab.parseToNode(text)
    while node:
        term = node.feature.split(",")[0]
        #print(f'{node.surface}:{term}')
        if term == "名詞":
            noun_list.append(node.surface)
        elif term == "動詞":
            verb_list.append(node.surface)
        elif term == "形容詞" or term == "副詞":
            adjv_list.append(node.surface)
        else:
            pass
        node = node.next
    
    for noun in noun_list:
        n[noun] += 1
            
    for verb in verb_list:
        v[verb] += 1
            
    for adjv in adjv_list:
        a[adjv] += 1
    
    return n, v, a

def main(args):
    csv.field_size_limit(sys.maxsize)
    DATASET = args.dataset
    LABEL = args.label
    MAX_LENGTH = args.length
    PATH = f'./tsv/{DATASET}/{LABEL}'
    SAVE_PATH = f'./morpheme/full/{DATASET}/{LABEL}'
    
    train = work.load_tsv_dataset(f'{PATH}/train.tsv')
    valid = work.load_tsv_dataset(f'{PATH}/dev.tsv')
    test  = work.load_tsv_dataset(f'{PATH}/test.tsv')
    works = train + valid + test
    
    st_time = time.time()
    
    print(f'num. of train: {len(train)}')
    print(f'num. of valid: {len(valid)}')
    print(f'num. of test : {len(test)}')
    print(f'num. of all  : {len(works)}')
    
    del(train)
    del(valid)
    del(test)
   
    pos_works = findPositive(works)
    
    print(f'num. of postive works: {len(pos_works)}')
    del(works)
    
    noun = defaultdict(int)
    verb = defaultdict(int)
    adjv = defaultdict(int)
    """
    noun = pd.DataFrame(columns = ['word', 'count'])
    verb = pd.DataFrame(columns = ['word', 'count'])
    adjv = pd.DataFrame(columns = ['word', 'count'])
    """
    for w in tqdm(pos_works):
        cnt = w['content']
        sep = re.split("。|」",cnt)
        
        for text in sep:
            noun, verb, adjv = analyze(noun, verb, adjv, text)
            
            #noun = OrderedDict(sorted(noun.items(), key=lambda x:x[1], reverse=True))
            #verb = OrderedDict(sorted(verb.items(), key=lambda x:x[1], reverse=True))
            #adjv = OrderedDict(sorted(adjv.items(), key=lambda x:x[1], reverse=True))
            """
            noun = noun.sort_values('count', ascending=False).reset_index(drop=True)
            verb = verb.sort_values('count', ascending=False).reset_index(drop=True)
            adjv = adjv.sort_values('count', ascending=False).reset_index(drop=True)
            
            
            if len(noun) > MAX_LENGTH:
                noun = dict(itertools.islice(noun.items(), MAX_LENGTH))
            if len(verb) > MAX_LENGTH:
                verb = dict(itertools.islice(verb.items(), MAX_LENGTH))
            if len(adjv) > MAX_LENGTH:
                adjv = dict(itertools.islice(adjv.items(), MAX_LENGTH))
            """
    
    anl_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-st_time))
    
    noun = OrderedDict(sorted(noun.items(), key=lambda x:x[1], reverse=True))
    verb = OrderedDict(sorted(verb.items(), key=lambda x:x[1], reverse=True))
    adjv = OrderedDict(sorted(adjv.items(), key=lambda x:x[1], reverse=True))
    
    #noun = dict(itertools.islice(noun.items(), MAX_LENGTH))
    #verb = dict(itertools.islice(verb.items(), MAX_LENGTH))
    #adjv = dict(itertools.islice(adjv.items(), MAX_LENGTH))
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(f'{SAVE_PATH}/noun.tsv', 'w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['word', 'count'])
        for key, val in noun.items():
            w.writerow([key, val])
    
    with open(f'{SAVE_PATH}/verb.tsv', 'w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['word', 'count'])
        for key, val in verb.items():
            w.writerow([key, val])
            
    with open(f'{SAVE_PATH}/adjv.tsv', 'w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['word', 'count'])
        for key, val in adjv.items():
            w.writerow([key, val])
    
    """    
    noun.to_csv(f'{SAVE_PATH}/noun.tsv', sep='\t', index=False)
    verb.to_csv(f'{SAVE_PATH}/verb.tsv', sep='\t', index=False)
    adjv.to_csv(f'{SAVE_PATH}/adjv.tsv', sep='\t', index=False)
    """
    
    now = datetime.now()
    res_str = f'info-{now.strftime("%Y-%m-%d")}'
    with open(f'{SAVE_PATH}/../{res_str}.jsonl', 'a+', encoding='utf-8') as file:
        str_time = now.strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "label": LABEL, 
            "date": str_time, 
            "time": anl_time, 
            "max_length": MAX_LENGTH
        }
        json.dump(result, file)
        file.write("\n")
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Morphological analysis")
    parser.add_argument('--dataset', type=str, default='front-512')
    parser.add_argument('--label', type=str, default='42')
    parser.add_argument('--length', type=int, default='512')
    args = parser.parse_args()
    
    print(f'Morphological Analysis\n    Dataset:{args.dataset}\n    Label:{args.label}')
    main(args)
