import torch
import json
from os import walk
from os import makedirs
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import BertJapaneseTokenizer
from torch.nn import Softmax

def findConfidence(label, paragraphs, bert, tokenizer):
    confidenceScores = []
    for paragraph in paragraphs:
        try:
            tokenized = tokenizer.tokenize(paragraph)
            tokenized.insert(0, '[CLS]')
            tokenized.append('[SEP]')
            tokens = tokenizer.convert_tokens_to_ids(tokenized)
            tensor_tokens = torch.tensor([tokens])
            output = bert(tensor_tokens)
            softmax = Softmax(dim=1)
            score = softmax(output[0]).cpu().detach().numpy()[0]
            if label == 1:
                confidenceScores.append(score[0])
            elif label == 0:
                confidenceScores.append(score[1])
        except RuntimeError as e:
            print(paragraph)
            print(e)
            
    confidenceParagraphs = []
    if len(confidenceScores) > 0:
        for i, score in enumerate(confidenceScores):
            confidenceParagraphs.append({
                'confidence':str(score), 
                'content':paragraphs[i]
            })
    else:
        return False, None

    status = {
        'label': label, 
        'confidence_scores': str(confidenceScores), 
        'contents': confidenceParagraphs
    }
    return True, status
        
            

def embeddingConfidence(works, target):        
    bert = torch.load(f'../savepoint/bert-fm/bert-fm-{target}.pt')
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    
    result = []
    for work in tqdm(works):
        label = work['label']
        paragraphs = [w['paragraph'] for w in work['contents']]
        try:
            success, status = findConfidence(label, paragraphs, bert, tokenizer)
        except TypeError as e:
            print(paragraphs)
            print(e)
        if success:
            result.append(status)
    return result

def main(args):
    TARGET = args.target
    WORK_DIR = f'../tsv/first-match-scatter/{TARGET}/'
    
    with open(f'{WORK_DIR}/test.json', 'r', encoding='utf-8') as f:
        works = json.load(f)
        
    confidence = embeddingConfidence(works, TARGET)
    
    with open(f'{WORK_DIR}/fm-confidence.json', 'w+', encoding='utf-8') as f:
        json.dump(confidence, f)
        
    
    
    

if __name__ == '__main__':
    parser = ArgumentParser(description="BERT confidence embedding")
    parser.add_argument('--target', type=int, default=42)
    args = parser.parse_args()
    
    print(f'Embedding BERT confidence\n\tLabel: {args.target}')
    
    main(args)