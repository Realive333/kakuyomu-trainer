import torch
import json
from os import walk
from os import makedirs
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import BertJapaneseTokenizer
from torch.nn import Softmax


def findConfidence(label, paragraphs, bert, tokenizer):
    confidentScores = []
    for paragraph in paragraphs[:10]:
        try:
            content = paragraph['content']
            tokenized = tokenizer.tokenize(content)
            #tokenized.insert(0, '[CLS]')
            #tokenized.append('[SEP]')
            tokens = tokenizer.convert_tokens_to_ids(tokenized)
            tensor_tokens = torch.tensor([tokens])
            output = bert(tensor_tokens)
            softmax = Softmax(dim=1)
            score = softmax(output[0]).cpu().detach().numpy()[0]
            if label == 1:
                confidentScores.append(score[0])
            elif label == 0:
                confidentScores.append(score[1])
        except RuntimeError as e:
            print(content)
            print(e)
    confidenceParagraphs = []
    if len(confidentScores) > 0:
        for i, score in enumerate(confidentScores):
            confidenceParagraphs.append({'confidence':str(score), 'score':paragraphs[i]['score'], 'content':paragraphs[i]['content']})
    else:
        return False, None
    status = {
        'label': label, 
        'confidence_scores': str(confidentScores), 
        'contents': confidenceParagraphs
    }
    return True, status

def traversalFiles(path, files, bert, tokenizer):
    result = []
    for file in tqdm(files):
        with open(f'{path}/{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data['label'] == '10':
            label = 1
        else:
            label = 0
        paragraphs = data['status']
        success, status = findConfidence(label, paragraphs, bert, tokenizer)
        if success:
            result.append(status)
    writeTarget = {
        'works': result
    }
    return writeTarget

def main(args):    
    TARGET = args.target
    
    ### NEED TO CHANGE BERT ###
    bert = torch.load(f'./savepoint/bert-fmfnb/bert-fmfnb-{TARGET}.pt')
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    
    savePath = f'./tsv/first-match-lexicon/{TARGET}/confidence/'
    makedirs(savePath, exist_ok=True)
    
    devPath = f'./tsv/nearest-k-candidates-offset-faster/n-100/o-256/{TARGET}/devs/'
    devFiles = next(walk(devPath), (None, None, []))[2]
    devWrite = traversalFiles(devPath, devFiles, bert, tokenizer)
    
    with open(f'{savePath}/dev_result.json', 'w+', encoding='utf-8') as f:
        json.dump(devWrite, f)
        
    testPath = f'./tsv/nearest-k-candidates-offset-faster/n-100/o-256/{TARGET}/test/'
    testFiles = next(walk(testPath), (None, None, []))[2]
    testWrite = traversalFiles(testPath, testFiles, bert, tokenizer)
    
    with open(f'{savePath}/test_result.json', 'w+', encoding='utf-8') as f:
        json.dump(testWrite, f)
    
    trainPath = f'./tsv/nearest-k-candidates-offset-faster/n-100/o-256/{TARGET}/train/'
    trainFiles = next(walk(trainPath), (None, None, []))[2]
    trainWrite = traversalFiles(trainPath, trainFiles, bert, tokenizer)
        
    with open(f'{savePath}/train_result.json', 'w+', encoding='utf-8') as f:
        json.dump(trainWrite, f)

if __name__ == '__main__':
    parser = ArgumentParser(description="Ideal Accuracy by All Candidates Test")
    parser.add_argument('--target', type=str, default='1')
    args = parser.parse_args()
    
    print(f'Testing Ideal Accuracy \n\tLabel: {args.target}')
    main(args)    
