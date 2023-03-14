import json
import torch
from tqdm import tqdm
from torch import nn
from transformers import BertJapaneseTokenizer
from argparse import ArgumentParser

parser = ArgumentParser(description="CLS classification")
parser.add_argument("--target", type=int, default=42)
parser.add_argument("--batchsize", type=int, default=5)
args = parser.parse_args()

print(f"CLS Classification\n\tLabel: {args.target}\n\tBatch Size: {args.batchsize}")

TARGET = args.target
BATCH_SIZE = args.batchsize

BERT = torch.load(f"../savepoint/bert-fm/bert-fm-{TARGET}.pt")
BERT = BERT.module.to('cuda')
BERT_WEIGHT = BERT.classifier.weight
BERT_BIAS = BERT.classifier.bias

bert_w_lst = BERT_WEIGHT.tolist()
grand_w_lst_l = []
grand_w_lst_r = []
for i in range(BATCH_SIZE):
    grand_w_lst_l += bert_w_lst[0]
    grand_w_lst_r += bert_w_lst[1]
BERT_WEIGHTED_WEIGHT = torch.tensor([grand_w_lst_l, grand_w_lst_r]).to('cuda')

TOKENIZER = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", return_tensors='pt', padding='max_length', max_length=1024)

def classification(work):
    tensor_ids = torch.tensor(work['batch']).to('cuda')
    out = BERT(tensor_ids, output_hidden_states=True)
    
    last_hidden_state = out.hidden_states[-1]
    pooler_output = BERT.bert.pooler(last_hidden_state)
    average = cls_average(pooler_output)
    grand = cls_grand(pooler_output)
    softmax = nn.Softmax(dim=-1)
    
    work_json = {
        'label': work['label'],
        'logits': out[0].tolist(),
        'avg': average.tolist(),
        'grand': grand.tolist(),
        'logits_softmax': [softmax(i).tolist() for i in out[0]],
        'avg_softmax': softmax(average).tolist(),
        'grand_softmax': softmax(grand).tolist()
    }
    return work_json

def cls_average(pooler_output):
    cls_sum = list(0.0 for i in range(768))
    for pool_cls in pooler_output:
        cls_sum = [a+b for a, b in zip(pool_cls, cls_sum)]
    cls_avg = torch.tensor([a/BATCH_SIZE for a in cls_sum]).to('cuda')
    
    avg_linear = nn.Linear(768, 2).to('cuda')
    nn.init.normal_(avg_linear.weight, std=0.0001)
    nn.init.normal_(avg_linear.bias, 0.01)
    
    return BERT.classifier(cls_avg)

def cls_grand(pooler_output):
    cls_append = []
    for pool_cls in pooler_output:
        cls_append += pool_cls
    cls_gnd = torch.tensor(cls_append).to('cuda')
    
    gnd_linear = nn.Linear(768*BATCH_SIZE, 2).to('cuda')
    nn.init.normal_(gnd_linear.weight, std=0.0001)
    nn.init.normal_(gnd_linear.bias, 0.01)
    #gnd_linear.weight = torch.nn.Parameter(BERT_WEIGHTED_WEIGHT)
    #gnd_linear.bias = BERT_BIAS
    
    return gnd_linear(cls_gnd)

def main(args):
    works_path = f"../tsv/first-match-scatter/{TARGET}/test.json"
    with open(works_path, "r") as f:
        works = json.load(f)
        
    work_results = []
    for work in tqdm(works):
        label = work['label']
        contents = [w['paragraph'] for w in work['contents']]
        if len(contents) != BATCH_SIZE:
            print(f"work should have batch size {BATCH_SIZE}, is {len(contents)}")
            continue
        batch = TOKENIZER.batch_encode_plus(contents, pad_to_max_length=True, max_length=512, truncation=True, add_special_tokens=True)
        iter_work = {'label': label, 'batch': batch['input_ids']}
        try:
            work_results.append(classification(iter_work))
        except RuntimeError as e:
            print(e)
            print(f"work should have batch size {BATCH_SIZE}, is {len(iter_work['batch'])}")
            
            
    save_path = f"../tsv/first-match-scatter/{TARGET}/cls-result-2.json"
    with open(save_path, "w+", encoding="utf-8") as f:
        json.dump(work_results, f)
    
if __name__ == '__main__':
    main(args)