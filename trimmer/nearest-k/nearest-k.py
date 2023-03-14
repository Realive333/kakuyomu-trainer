import os
import re
import csv
import sys
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
        if scoreCount >= scores[0]['score'] and wordCount >= 512 and scoreCount > 0:
            scores.insert(0, {'pos': positionCount, 'len': wordCount, 'end': endCount, 'score': scoreCount})
    return scores

def savePositionScoreTSV(works, simKList, savePath):
    results = []
    for work in tqdm(works): # TEST
        if work['label'] == 0:
            label = '01'
        else:
            label = '10'
        
        content = work['content']
        content = content.replace('\n', '')
        sep = re.split(" ", content)
        
        score = getPositionScore(simKList, sep)[0]
        
        if score['score'] > 0:
            paragraph = ' '.join(sep[score['pos']:score['end']])
        else:
            paragraph = content[:512]
        result = {'label': label, 'content': paragraph, 'score': score}
        results.append(result)
    
    with open(savePath, 'w+', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        for result in results:
            writer.writerow([result['label'], result['content']])
    
    with open(f'{savePath}_status.tsv', 'w+', encoding='utf-8') as f2:
        writer = csv.writer(f2, delimiter='\t')
        for result in results:
            writer.writerow([result['label'], result['score']])

def main(args):
    target = args.target
    kSize = args.ksize
    
    SIM_PATH = f'./morpheme/similarity/{target}'
    WORK_PATH = f'./tsv/morpheme/{target}'
    SAVE_PATH = f'./tsv/nearest-k_kai/k-{kSize}/{target}'

    trains = wf.load_tsv_dataset(f'{WORK_PATH}/train.tsv')
    tests = wf.load_tsv_dataset(f'{WORK_PATH}/test.tsv')
    devs = wf.load_tsv_dataset(f'{WORK_PATH}/dev.tsv')
    
    simList = wf.load_tsv_similarity(f'{SIM_PATH}/total_avg.tsv')
    simKList = [word['word'] for word in simList][:kSize]
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    savePositionScoreTSV(trains, simKList, f'{SAVE_PATH}/train.tsv')
    savePositionScoreTSV(tests, simKList, f'{SAVE_PATH}/test.tsv')
    savePositionScoreTSV(devs, simKList, f'{SAVE_PATH}/dev.tsv')
    
if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    parser = ArgumentParser(description="Nearest K TSV")
    parser.add_argument('--target', type=str, default='1')
    parser.add_argument('--ksize', type=int, default=10)
    args = parser.parse_args()
    
    print(f'Create Nearest-{args.ksize} TSV\n\tLabel: {args.target}')
    main(args)    
