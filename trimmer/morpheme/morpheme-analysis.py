import os
import sys
import csv
import MeCab
import pandas as pd
import work
import time
import json

from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser

def analyze(n, v, a, text):
    noun_list = []
    verb_list = []
    adjv_list = []
    mecab = MeCab.Tagger("")
    
    node = mecab.parseToNode(text)
    while node:
        term = node.feature.split(",")[0]
        #print(f'{node.surface}:{term}')
        if term == "名詞":
            noun_list.append(node.surface)
        elif term == "動詞":
            verb_list.append(node.surface)
        elif term == "形容詞" or term == "副詞":
            adjv_list.append(node.surface)
        else:
            pass
        node = node.next
    
    for noun in noun_list:
        if noun in n.values:
            c = n.loc[n['word']==noun]['count']
            n.loc[n.word==noun, 'count'] = c.values[0]+1
        else:
            r = pd.DataFrame({'word':noun, 'count':1}, index=[0])
            n = pd.concat([r, n], ignore_index=True)
            
    for verb in verb_list:
        if verb in v.values:
            c = v.loc[v['word']==verb]['count']
            v.loc[v.word==verb, 'count'] = c.values[0]+1
        else:
            r = pd.DataFrame({'word':verb, 'count':1}, index=[0])
            v = pd.concat([r, v], ignore_index=True)
            
    for adjv in adjv_list:
        if adjv in a.values:
            c = a.loc[a['word']==adjv]['count']
            a.loc[a.word==adjv, 'count'] = c.values[0]+1
        else:
            r = pd.DataFrame({'word':adjv, 'count':1}, index=[0])
            a = pd.concat([r, a], ignore_index=True)
    
    return n, v, a

def main(args):
    csv.field_size_limit(sys.maxsize)
    path = f'./tsv/{args.dataset}/{args.target}'
    train = work.load_tsv_dataset(f'{path}/train.tsv')
    valid = work.load_tsv_dataset(f'{path}/dev.tsv')
    test = work.load_tsv_dataset(f'{path}/test.tsv')
    works = train + valid + test
    
    st_time = time.time()
    
    print(f'num. of train: {len(train)}')
    print(f'num. of valid: {len(valid)}')
    print(f'num. of test : {len(test)}')
    print(f'num. of all  : {len(works)}')
    
    del(train)
    del(valid)
    del(test)
    
    pos_works = [work for work in works if not work['label']==0]
    for w in pos_works:
        if w['label'] == 0:
            raise Exception('All works should be positive example, has', work)

    print(f'num. of postive: {len(pos_works)}')
    
    noun = pd.DataFrame(columns = ['word', 'count'])
    verb = pd.DataFrame(columns = ['word', 'count'])
    adjv = pd.DataFrame(columns = ['word', 'count'])

    for w in tqdm(pos_works):
        sep = w['content'].split("。")
        for text in sep:
            noun, verb, adjv = analyze(noun, verb, adjv, text)
            
    anl_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-st_time))
            
    noun = noun.sort_values('count', ascending=False).reset_index(drop=True)
    verb = verb.sort_values('count', ascending=False).reset_index(drop=True)
    adjv = adjv.sort_values('count', ascending=False).reset_index(drop=True)
    
    t_path = f'./morpheme/{args.dataset}/{args.target}'
    os.makedirs(t_path, exist_ok=True)
    noun.to_csv(f'{t_path}/noun.tsv', sep='\t', index=False)
    verb.to_csv(f'{t_path}/verb.tsv', sep='\t', index=False)
    adjv.to_csv(f'{t_path}/adjv.tsv', sep='\t', index=False)
    
    now = datetime.now()
    res_str = f'info-{now.strftime("%Y-%m-%d")}'
    with open(f'{t_path}/../{res_str}.jsonl', 'a+', encoding='utf-8') as file:
        str_time = now.strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "label": args.target, 
            "date": str_time, 
            "time": anl_time, 
            "noun count": len(noun), 
            "verb count": len(verb), 
            "adjv count": len(adjv)
        }
        json.dump(result, file)
        file.write("\n")

if __name__ == "__main__":
    parser = ArgumentParser(description="Morpheme Analysis")
    parser.add_argument('--dataset', type=str, default='front-512')
    parser.add_argument('--target', type=str, default='1')
    args = parser.parse_args()
    
    print(f'Morpheme Analysis\n    DATASET: {args.dataset}\n    LABEL:{args.target}')
    main(args)
