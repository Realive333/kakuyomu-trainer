import os
import re
import csv
import sys
import json
import workFactory as wf

from tqdm import tqdm
from argparse import ArgumentParser

def getPositionScore(matchWords, sep):
    # define variables
    scores = [{'pos': 0, 'len': 0, 'end': 0, 'score':0}]
    wordCount = 0
    scoreCount = 0
    positionCount = 0
    endCount = 0
    
    # calculate position scores
    while endCount < len(sep):
        if wordCount < 512:
            wordCount += len(sep[endCount])
            if sep[endCount] in matchWords:
                scoreCount += 1 # we can change weight by setting into word score
        elif wordCount == 512:
            wordCount -= len(sep[positionCount])
            if sep[positionCount] in matchWords:
                scoreCount -= 1 # we can change weight by setting into word score
            wordCount += len(sep[endCount])
            if sep[endCount] in matchWords:
                scoreCount += 1 # we can change weight by setting into word score
            positionCount += 1
        elif wordCount > 512:
            wordCount -= len(sep[positionCount])
            if sep[positionCount] in matchWords:
                scoreCount -= 1 # we can change weight by setting into word score
            positionCount += 1
            endCount -= 1
        endCount += 1
        if wordCount >= 512 and scoreCount > 5:    ### A FIXED NUMBER ###
            scores.insert(0, {'pos': positionCount, 'len': wordCount, 'end': endCount, 'score': scoreCount})
    return scores

def getNearestKCandidates(works, simKList, savePath):
    for i, work in enumerate(tqdm(works)):
        content = work['content']
        content = content.replace('\n', '')
        sep = re.split(" ", content)
        scores = getPositionScore(simKList, sep)
        
        candidates = [scores[0]]
        for score in scores:
            if score['end'] < candidates[0]['pos'] and score['score'] >= 0:
                candidates.insert(0, score)
        
        if candidates[0]['score'] == 0:
            candidates.pop(0)

        candidateList = []
        if len(candidates)!= 0:
            for candidate in candidates:
                candidateList.append({
                    'pos': candidate['pos'],
                    'len': candidate['len'],
                    'end': candidate['end'],
                    'score': candidate['score'],
                    'content': ' '.join(sep[candidate['pos']:candidate['end']])
                })
        else:
            candidateList.append({
                'pos': 0,
                'len': 512,
                'end': 512,
                'score': 0,
                'content': content[:512]
            })
        
        if work['label'] == 0:
            label = '01'
        else:
            label = '10'
        
        writeTarget = {
            'label': label,
            'status': candidateList[slice(None, None, -1)]
        }
    
        with open(f'{savePath}/{i}.json', 'w+', encoding='utf-8') as f:
            json.dump(writeTarget, f)
            
def main(args):
    target = args.target
    kSize = args.ksize
    
    SIM_PATH = f'./morpheme/similarity/{target}'
    WORK_PATH = f'./tsv/morpheme/{target}'
    SAVE_PATH = f'./tsv/nearest-k-candidates-allscore/n-{kSize}/{target}'

    trains = wf.load_tsv_dataset(f'{WORK_PATH}/train.tsv')
    tests = wf.load_tsv_dataset(f'{WORK_PATH}/test.tsv')
    devs = wf.load_tsv_dataset(f'{WORK_PATH}/dev.tsv')
    
    simList = wf.load_tsv_similarity(f'{SIM_PATH}/total_avg.tsv')
    simKList = [word['word'] for word in simList][:kSize]
    
    os.makedirs(f'{SAVE_PATH}/train', exist_ok=True)
    getNearestKCandidates(trains, simKList, f'{SAVE_PATH}/train')
    
    os.makedirs(f'{SAVE_PATH}/test', exist_ok=True)
    getNearestKCandidates(tests, simKList, f'{SAVE_PATH}/test')
    
    os.makedirs(f'{SAVE_PATH}/dev', exist_ok=True)
    getNearestKCandidates(devs, simKList, f'{SAVE_PATH}/dev')
    
if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    parser = ArgumentParser(description="Nearest K Candidates")
    parser.add_argument('--target', type=str, default='1')
    parser.add_argument('--ksize', type=int, default=10)
    args = parser.parse_args()
    
    print(f'Create Nearest-{args.ksize} Candidates TSV\n\tLabel: {args.target}')
    main(args)    
