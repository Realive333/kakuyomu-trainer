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

def getJointListByOffset(content, offset):
    content = content.replace("\n", "\n ")
    sep = re.split(" ", content)
    sepList = [] 
    compList = []
    for i, word in enumerate(sep):
        if sum(len(i) for i in sepList) < offset:
            sepList.append(word)
            if i == len(sep)-1:
                compList.append(" ".join(sepList))
        else:
            compList.append(" ".join(sepList))
            sepList.clear()
            sepList.append(word)
    jointList = []
    for i, _ in enumerate(compList):
        try:
            if offset == 256:
                jointList.append(compList[i]+compList[i+1])
            elif offset == 512:
                jointList.append(compList[i])
        except IndexError:
            _
    return jointList

def getPScoreOffset(simKList, content, offset, mecab):
    jointList = getJointListByOffset(content, offset)
    result = []
    for joint in jointList:
        paragraph = joint.replace("\n", "[SEP]")
        sep = re.split(" ", paragraph)
        score = 0
        for word in sep:
            if word in simKList:
                score += 1
        if score != 0:
            result.append({
                'score': score,
                'content': " ".join(sep)
            })
    
    resultNum = len(result)
    if resultNum < 10:
        extra = 10 - resultNum
        for i in range(0, extra):
            if len(content) > 512:
                rand = random.randint(0, len(content)-512)
            else:
                rand = 0
            randContent = content[rand:]
            randContent = randContent.replace(" ", "")
            randContent = randContent.replace("\n", "\\")
            randContent = randContent[:512]
            randResult = mecab.parse(randContent)
            randResult = randResult.replace("\n", "")
            randResult = randResult.replace("\\", "[SEP]")
            result.append({
                'score': -1,
                'content': randResult
            })
    return resultNum, result
    

def getNKCandidatesOffset(works, simKList, offset, savePath, mecab):
    os.makedirs(savePath, exist_ok=True)
    for i, work in enumerate(tqdm(works)):
        content = work['content']
        
        resultNum, scores = getPScoreOffset(simKList, content, offset, mecab)
        if work['label'] == 0:
            label = '01'
        else:
            label = '10'
            
        writeTarget = {
            'label': label,
            'match_num': resultNum,
            'status': sorted(scores, key=lambda x: x['score'], reverse=True)
        }
    
        with open(f'{savePath}/{i}.json', 'w+', encoding='utf-8') as f:
            json.dump(writeTarget, f)
    
def main(args):
    target = args.target
    kSize = args.ksize
    offset = args.offset
    
    SIM_PATH = f'./morpheme/similarity/{target}'
    WORK_PATH = f'./tsv/morpheme/{target}'
    SAVE_PATH = f'./tsv/nearest-k-candidates-offset-faster/n-{kSize}/o-{offset}/{target}'
    
    trains = wf.load_tsv_dataset(f'{WORK_PATH}/train.tsv')
    tests = wf.load_tsv_dataset(f'{WORK_PATH}/test.tsv')
    devs = wf.load_tsv_dataset(f'{WORK_PATH}/dev.tsv')
    
    simList = wf.load_tsv_similarity(f'{SIM_PATH}/total_avg.tsv')
    simKList = [word['word'] for word in simList][:kSize]
    
    mecab = MeCab.Tagger("-Owakati")
    
    getNKCandidatesOffset(trains, simKList, offset, f'{SAVE_PATH}/train', mecab)
    getNKCandidatesOffset(tests, simKList, offset, f'{SAVE_PATH}/test', mecab)
    getNKCandidatesOffset(devs, simKList, offset, f'{SAVE_PATH}/devs', mecab)

if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    parser = ArgumentParser(description="Nearest K Candidates Offset")
    parser.add_argument('--target', type=str, default='1')
    parser.add_argument('--ksize', type=int, default=10)
    parser.add_argument('--offset', type=int, default=256)
    args = parser.parse_args()
    print(f'Create Nearest-{args.ksize} Candidates TSV FASTER\n\toffset: {args.offset}\n\ttarget: {args.target}')
    main(args)