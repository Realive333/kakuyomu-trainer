import os
import re
import csv
import sys
import json
import random
import MeCab
import workFactory as wf

from tqdm import tqdm
from argparse import ArgumentParser

def getPScoreOffset(simKList, content, offset, mecab):
    stageCount = len(content)//offset
    result = []
    for i in range(0, stageCount):
        paragraph = content[i*offset:i*offset+512]
        phrase = mecab.parse(paragraph).replace("\n", "")
        sep = re.split(" ", phrase)
        score = 0
        for word in sep:
            if word in simKList:
                score += 1
        if score != 0:
            result.append({
                'pos': i*offset,
                'end': i*offset+512,
                'score': score,
                'content': paragraph
                #'content': phrase
            })
            
    if len(result) < 10:
        extra = 10 - len(result)
        for i in range(0, extra):
            if len(content) > 512:
                rand = random.randint(0, len(content)-512)
            else:
                rand = 0
            result.append({
                'pos':rand, 
                'end':rand+512,
                'score': 0,
                'content': content[rand:rand+512]
                #'content':mecab.parse(content[:512]).replace("\n", "")
            })
    return result
    

def getNKCandidatesOffset(works, simKList, offset, savePath, mecab):
    os.makedirs(savePath, exist_ok=True)
    for i, work in enumerate(tqdm(works)):
        content = work['content']
        content = content.replace('\n', '')
        
        scores = getPScoreOffset(simKList, content, offset, mecab)
        if work['label'] == 0:
            label = '01'
        else:
            label = '10'
            
        writeTarget = {
            'label': label,
            'status': sorted(scores, key=lambda x: x['score'], reverse=True)
        }
    
        with open(f'{savePath}/{i}.json', 'w+', encoding='utf-8') as f:
            json.dump(writeTarget, f)
    
def main(args):
    target = args.target
    kSize = args.ksize
    offset = args.offset
    
    SIM_PATH = f'./morpheme/similarity/{target}'
    WORK_PATH = f'./tsv/full/{target}'
    SAVE_PATH = f'./tsv/nearest-k-candidates-offset/n-{kSize}/o-{offset}/{target}'
    
    trains = wf.load_tsv_dataset(f'{WORK_PATH}/train.tsv')
    tests = wf.load_tsv_dataset(f'{WORK_PATH}/test.tsv')
    devs = wf.load_tsv_dataset(f'{WORK_PATH}/dev.tsv')
    
    simList = wf.load_tsv_similarity(f'{SIM_PATH}/total_avg.tsv')
    simKList = [word['word'] for word in simList][:kSize]
    
    mecab = MeCab.Tagger("-Owakati")
    
    getNKCandidatesOffset(trains, simKList, offset, f'{SAVE_PATH}/trains', mecab)
    getNKCandidatesOffset(tests, simKList, offset, f'{SAVE_PATH}/test', mecab)
    getNKCandidatesOffset(devs, simKList, offset, f'{SAVE_PATH}/devs', mecab)

if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    parser = ArgumentParser(description="Nearest K Candidates Offset")
    parser.add_argument('--target', type=str, default='1')
    parser.add_argument('--ksize', type=int, default=10)
    parser.add_argument('--offset', type=int, default=256)
    args = parser.parse_args()
    print(f'Create Nearest-{args.ksize} Candidates TSV\n\toffset: {args.offset}\n\ttarget: {args.target}')
    main(args)