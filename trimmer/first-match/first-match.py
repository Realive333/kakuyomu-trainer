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

def write_df_to_tsv(path, dataframe, WORK_DIR):
    try:
        os.makedirs(WORK_DIR)
    except FileExistsError:
        print("folder exist")
        
    with open(path, 'w+', encoding='utf-8') as writer:
        tsv_writer = csv.writer(writer, delimiter='\t')
        for i, row in dataframe.iterrows():
            tsv_writer.writerow([row['label'], row['content']])

def findFirstMatchParagraph(work, label, pos, neg):
    labels = workFactory.load_tsv_label('../../numeric_label.tsv')
    match_list = labels[int(label)-1]['name']
    sep = re.split("。|」", work['content'])
    posit = -1
    conb = ''
    for i, sen in enumerate(sep):
        for match_word in match_list:
            if match_word in sen:
                posit = i
                break
    if posit > 0:
        conb = '。'.join(sep[posit:posit+100])
        conb = conb[:512]
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
    
    fp_works = []
    pos_match_count = 0
    neg_match_count = 0
    #final_works = []
    
    train = workFactory.load_tsv_dataset(f'{PATH}/train.tsv')
    valid = workFactory.load_tsv_dataset(f'{PATH}/dev.tsv')
    test = workFactory.load_tsv_dataset(f'{PATH}/test.tsv')
    works = train + valid + test
    """
    pos_works = [work for work in works if not work['label']==0]
    neg_works = [work for work in works if not work['label']==1]
    """
    del(train)
    del(valid)
    del(test)
    
    for work in tqdm(works):
        par, pos_match_count, neg_match_count = findFirstMatchParagraph(work, LABEL, pos_match_count, neg_match_count)
        fp_works.append(par)
    
    total_count = len(works)
    match_count = pos_match_count + neg_match_count
    posit_count = len([work for work in works if not work['label']==0])
    """    
    fp_length = len(fp_works)
    for work in fp_works:
        dat = {"label": "10", "content": work}
        final_works.append(dat)
    
    neg_works = neg_works[:fp_length]
    for work in neg_works:
        cnt = work['content']
        if len(cnt)<512:
            dat = {"label": "01", "content": cnt}
        else:
            rnd = randint(0, len(cnt)-512)
            dat = {"label": "01", "content": cnt[rnd:rnd+512]}
        final_works.append(dat)
    """
    final_df = pd.DataFrame(fp_works)
    train, valid_test = train_test_split(final_df, test_size=0.2)
    valid, test = train_test_split(valid_test, test_size=0.5)

    print(f'Train: {len(train)} / Valid: {len(valid)} / Test: {len(test)}')

    del(final_df)
    del(valid_test)
    
    write_df_to_tsv(f'{WORK_DIR}/train.tsv', train, WORK_DIR)
    write_df_to_tsv(f'{WORK_DIR}/dev.tsv', valid, WORK_DIR)
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