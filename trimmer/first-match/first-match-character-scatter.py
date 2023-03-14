import os 
import csv
import os
import csv
import sys
sys.path.append('../..')

import json
import random
import workFactory as wf

from tqdm import tqdm
from argparse import ArgumentParser

csv.field_size_limit(sys.maxsize)

def seperate(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def firstMatch_chunk(chunk, matchWords):
    for index, word in enumerate(chunk):
        if word in matchWords:
            l_paragraph = chunk[:index]
            r_paragraph = chunk[index:]
            ct = 0
            paragraph = []
            for wd in r_paragraph:
                ct += len(wd)
                paragraph.append(wd)
                if ct > 512:
                    return 'success at r_list', 'r', ' '.join(paragraph[:-1])
            if ct < 512:
                for wd in reversed(l_paragraph):
                    ct += len(wd)
                    paragraph.insert(0, wd)
                    if ct > 512:
                        return 'success at l_list', 'l', ' '.join(paragraph[1:])
    ct = 0
    for idx, wd in enumerate(chunk):
        ct += len(wd)
        if ct > 512:
            return 'no match', 'n', ' '.join(chunk[:idx])

def firstMatch_work(work, matchWords, sepNum):
    paragraphs = []
    split = work['content'].replace('\n', '').split()
    chunks = list(seperate(split, len(split)//sepNum))
    for chunk in chunks[:sepNum]:
        try:
            s, c, p = firstMatch_chunk(chunk, matchWords)
            paragraphs.append({'status': s, 'code': c, 'paragraph': p})
        except TypeError as err:
            print('Error at firstMatch_work', err)
            print(''.join(chunk))
    return paragraphs

def saveFMtoTSV_chunk(works, matchWords, savePath, sepNum):
    writeTarget = []
    for work in tqdm(works):
        try:
            wlen = len(work['content'].replace('\n', '').replace(' ', ''))
            if wlen > sepNum*512:
                if work['label'] == 0:
                    label = '01'
                else:
                    label = '10'

                paragraphs = firstMatch_work(work, matchWords, sepNum)
                for p in paragraphs:
                    writeTarget.append({
                        'label': label,
                        'content': p['paragraph']
                    })
        except ValueError as err:
            print('Error at FM2TSV', err)
            print(work)
    with open(savePath, 'w+', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        for target in tqdm(writeTarget):
            writer.writerow([target['label'], target['content']])

def saveFMtoJSON_chunk(works, matchWords, savePath, sepNum):
    writeTarget = []
    for work in tqdm(works):
        try:
            wlen = len(work['content'].replace('\n', '').replace(' ', ''))
            if wlen > sepNum*512:
                label = work['label']
                paragraphs = firstMatch_work(work, matchWords, sepNum)
                writeTarget.append({
                    'label': label,
                    'contents': paragraphs
                })
        except ValueError as err:
            print('Error at FM2JSON', err)
            print(work)
    with open(f'{savePath}', 'w+', encoding='utf-8') as f:
        json.dump(writeTarget, f)
        
def main(args):
    TARGET = args.target
    NUM = args.num
    MODE = args.mode
    
    WORK_DIR = f'../../tsv/morpheme/{TARGET}'
    SAVE_DIR = f'../../tsv/first-match-scatter/{TARGET}'
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    labels = wf.load_tsv_label('../../numeric_label.tsv')
    matchWords = labels[TARGET-1]['name']
    
    train = wf.load_tsv_dataset(f'{WORK_DIR}/train.tsv')
    dev = wf.load_tsv_dataset(f'{WORK_DIR}/dev.tsv')
    test = wf.load_tsv_dataset(f'{WORK_DIR}/test.tsv')
    
    if MODE == 'json':
        saveFMtoJSON_chunk(train, matchWords, f'{SAVE_DIR}/train.json', NUM)
        saveFMtoJSON_chunk(dev, matchWords, f'{SAVE_DIR}/dev.json', NUM)
        saveFMtoJSON_chunk(test, matchWords, f'{SAVE_DIR}/test.json', NUM)
    elif MODE == 'tsv':
        saveFMtoTSV_chunk(train, matchWords, f'{SAVE_DIR}/train.tsv', NUM)
        saveFMtoTSV_chunk(dev, matchWords, f'{SAVE_DIR}/dev.tsv', NUM)
        saveFMtoTSV_chunk(test, matchWords, f'{SAVE_DIR}/test.tsv', NUM)

if __name__ == "__main__":
    parser = ArgumentParser(description="First Match Character Scatter")
    parser.add_argument('--target', type=int, default=42)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--mode', type=str, default='json')
    
    args = parser.parse_args()
    
    print(f'create first-match_character-scatter\n\tLabel: {args.target}\n\tNum: {args.num}')
    main(args)