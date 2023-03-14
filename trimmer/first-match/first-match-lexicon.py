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

def firstMatch_Lexicon(work, matchWords):
    split = work['content'].replace('\n', '').split()
    paragraphs = []
    
    for index, word in enumerate(split):
        if word in matchWords:
            paragraph = split[index:index+512]
            paragraphs.append(' '.join(paragraph))
    
    matchNum = len(paragraphs)
    if matchNum >= 10:
        paragraphs = paragraphs[:10]
    else:
        diff = 10-matchNum
        for i in range(diff):
            if len(split) > 512:
                rand = random.randint(0, len(split)-512)
            else:
                rand = 0
            paragraphs.append(' '.join(split[rand:rand+512]))     
    return matchNum, paragraphs

def saveFM_Lexicon(works, matchWords, savePath):
    writeTarget = []
    for work in tqdm(works):
        label = work['label']
        num, paragraphs = firstMatch_Lexicon(work, matchWords)
        writeTarget.append({
            'label': label,
            'match': num,
            'contents': paragraphs
        })
    with open(f'{savePath}', 'w+', encoding='utf-8') as f:
        json.dump(writeTarget, f)

def main(args):
    TARGET = args.target
    WORK_PATH = f'../../tsv/morpheme/{TARGET}'
    SAVE_DIR = f'../../tsv/first-match-lexicon/{TARGET}'
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    labels = wf.load_tsv_label('../../numeric_label.tsv')
    matchWords = labels[TARGET-1]['name']
    
    train = wf.load_tsv_dataset(f'{WORK_PATH}/train.tsv')
    saveFM_Lexicon(train, matchWords, f'{SAVE_DIR}/train.json')
    
    dev = wf.load_tsv_dataset(f'{WORK_PATH}/dev.tsv')
    saveFM_Lexicon(dev, matchWords, f'{SAVE_DIR}/dev.json')
    
    test = wf.load_tsv_dataset(f'{WORK_PATH}/test.tsv')
    saveFM_Lexicon(test, matchWords, f'{SAVE_DIR}/test.json')
    

if __name__ == "__main__":
    parser = ArgumentParser(description="First Match Lexicon")
    parser.add_argument('--target', type=int, default=42)
    
    args = parser.parse_args()
    
    print(f'create first-match_lexicon\n\tLabel: {args.target}')
    main(args)