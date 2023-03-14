import workFactory
import json
import csv
import sys
import re
import os
import pandas as pd

from tqdm import tqdm
from random import randint
from datetime import datetime
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

def write_df_to_tsv(path, works, WORK_DIR):
    try:
        os.makedirs(WORK_DIR)
    except FileExistsError:
        print("folder exist")
        
    with open(path, 'w+', encoding='utf-8') as writer:
        tsv_writer = csv.writer(writer, delimiter='\t')
        for work in works:
            tsv_writer.writerow([work['label'], work['content']])
            
def findMatchPosition(work, label, sep):
    labels = workFactory.load_tsv_label('../../numeric_label.tsv')
    match_list = labels[int(label)-1]['name']
    posit = -1
    conb = ''
    for i, sen in enumerate(sep):
        for match_word in match_list:
            if match_word in sen:
                posit = i
                return posit
    return posit
    
def findFirstMatchParagraph(work, label, pos, neg):
    sep = re.split("。|！|？|\\?|\\!|\\n", work['content'])
    
    posit = findMatchPosition(work, label, sep)
    
    if posit > 0:
        front = '。'.join(sep[posit:])
        conb = front[:512]
        """
        front = '。'.join(sep[:posit])
        back = '。'.join(sep[posit:])
        wFront = front[-255:]
        wBack = back[:256]
        
        if len(wFront) < 255:
            offset = 255 - len(wFront)
            wBack = back[:256 + offset]
            
        if len(wBack) < 256:
            preset = 256 - len(wBack)
            wFront = front[-(255+preset):]
        
        conb = wFront + '。' + wBack
        if len(conb) != 512:
            print(f'Length != 512: {len(wFront)} : {len(wBack)}')
            print(conb)
        """
        
        """
        cnt = 1
        while 1:
            if posit - cnt > 0:
                front = posit - cnt
            else:
                front = posit
            back = posit + cnt
            
            front =  '。'.join(sep[front:posit])
            back  =  '。'.join(sep[posit:back])
            conb = front + back
            
            if len(conb) > 512:
                if len(conb) > 1024:
                    front = front[-512:]
                    back = back[:512]
                    conb = front + back
                    print(conb)
                break
            elif cnt > 100:
                break
            cnt += 1
        """
        if work['label'] == 0:
            label = '01'
            neg += 1
        else:
            label = '10'
            pos += 1
    else:
        conb = '。'.join(sep[:100])
        conb = conb[:512]
        if work['label'] == 0:
            label = '01'
        else:
            label = '10'
    
    
    return {"label": label, "content": conb}, pos, neg
            
def main(args):
    csv.field_size_limit(sys.maxsize)
    DATASET = args.dataset
    LABEL = args.target
    PATH = f'../../tsv/{DATASET}/{LABEL}'
    WORK_DIR = f'../../tsv/first-match/{LABEL}'
    
    
    pos_match_count = 0
    neg_match_count = 0
    #final_works = []
    
    fTrain = workFactory.load_tsv_dataset(f'{PATH}/train.tsv')
    fDev = workFactory.load_tsv_dataset(f'{PATH}/dev.tsv')
    fTest = workFactory.load_tsv_dataset(f'{PATH}/test.tsv')
    works = fTrain + fDev + fTest
    
    train = []
    dev = []
    test = []
    ### TEST ###
    
    for work in tqdm(fTrain):
        result, pos_match_count, neg_match_count = findFirstMatchParagraph(work, LABEL, pos_match_count, neg_match_count)
        train.append(result)
    
    for work in tqdm(fDev):
        result, pos_match_count, neg_match_count = findFirstMatchParagraph(work, LABEL, pos_match_count, neg_match_count)
        dev.append(result)
    
    for work in tqdm(fTest):
        result, pos_match_count, neg_match_count = findFirstMatchParagraph(work, LABEL, pos_match_count, neg_match_count)
        test.append(result)
    """
    for work in tqdm(works):
        par, pos_match_count, neg_match_count = findFirstMatchParagraph(work, LABEL, pos_match_count, neg_match_count)
        fp_works.append(par)
    
    """
    total_count = len(works)
    match_count = pos_match_count + neg_match_count
    posit_count = len([work for work in works if not work['label']==0])
    
    #final_df = pd.DataFrame(fp_works)
    #train, valid_test = train_test_split(final_df, test_size=0.2)
    #valid, test = train_test_split(valid_test, test_size=0.5)

    print(f'Train: {len(train)} / dev: {len(dev)} / Test: {len(test)}')

    #del(final_df)
    #del(valid_test)
    
    write_df_to_tsv(f'{WORK_DIR}/train.tsv', train, WORK_DIR)
    write_df_to_tsv(f'{WORK_DIR}/dev.tsv', dev, WORK_DIR)
    write_df_to_tsv(f'{WORK_DIR}/test.tsv', test, WORK_DIR)
    
    now = datetime.now()
    str_date = now.strftime('%Y-%m-%d')
    with open(f'{WORK_DIR}/{str_date}-record.json', 'w', encoding="utf-8") as f:
        record = {
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "total works": total_count,
            "matched works": match_count,
            "positive works": posit_count,
            "positive match works": pos_match_count,
            "negative match works": neg_match_count
        }
        json.dump(record, f)
        

if __name__ == "__main__":
    parser = ArgumentParser(description="First Match TSV")
    parser.add_argument('--dataset', type=str, default='full')
    parser.add_argument('--target', type=str, default='1')
    args = parser.parse_args()
    
    print(f'create first-match tsv\n\tDataset:{args.dataset}\n\tLabel:{args.target}')
    main(args)
