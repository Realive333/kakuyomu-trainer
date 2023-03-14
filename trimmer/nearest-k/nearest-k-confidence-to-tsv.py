import re
import os
import csv
import sys
import json

from tqdm import tqdm
from argparse import ArgumentParser

def appendContentsToTSV(work, savepath):
    result = []
    contents = work['contents']
    for content in contents:
        confidence = float(content['confidence'])
        score = content['score']
        paragraph = content['content']
        paragraph = paragraph.replace("[SEP]", " ")
        result.append({'confidence': confidence, 'score': score, 'content': paragraph})
    with open(savepath, 'a', encoding='UTF-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t', lineterminator='\n')
        for r in result:
            writer.writerow([r['confidence'], r['score'], r['content']])

def traversalWorks(works, filePath):
    if os.path.isfile(filePath):    # Remove file if exist, to prevent duplicate appending
        os.remove(filePath)    
    for work in tqdm(works):
        appendContentsToTSV(work, filePath)
            
def main(args):
    target = args.target
    kSize = args.ksize
    offset = args.offset
    
    WORK_PATH = f'./tsv/nearest-k-candidates-offset-faster/n-{kSize}/o-{offset}/{target}/confidence/top-10'
    SAVE_PATH = f'./tsv/nearest-k-confidence-BERT/n-{kSize}/o-{offset}/{target}'

    os.makedirs(SAVE_PATH, exist_ok=True)
    
    ### TRAIN ###
    with open(f'{WORK_PATH}/train_result.json') as f:
        trainWorks = json.load(f)
    trainDir = f'{SAVE_PATH}/train.tsv'
    traversalWorks(trainWorks['works'], trainDir)
    
    ### DEV ###
    with open(f'{WORK_PATH}/dev_result.json') as f:
        devWorks = json.load(f)
    devDir = f'{SAVE_PATH}/dev.tsv'
    traversalWorks(devWorks['works'], devDir)
    
    ### TEST ###
    with open(f'{WORK_PATH}/test_result.json') as f:
        testWorks = json.load(f)
    testDir = f'{SAVE_PATH}/test.tsv'
    traversalWorks(testWorks['works'], testDir)
    
if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    
    parser = ArgumentParser(description="Nearest K Confidence Write TSV")
    parser.add_argument('--target', type=str, default='1')
    parser.add_argument('--ksize', type=int, default=100)
    parser.add_argument('--offset', type=int, default=256)
    args = parser.parse_args()

    print(f'Create Nearest-{args.ksize} Confidence TSV\n\toffset: {args.offset}\n\ttarget: {args.target}')
    main(args)