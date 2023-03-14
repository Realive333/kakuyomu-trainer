import os
import csv
import sys
import json
import work
import pandas as pd
import MeCab
import time
import pickle

from argparse import ArgumentParser
from tqdm import tqdm

from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
### MODELS ###
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


def main(args):
    path = f'../tsv/{args.dataset}/{args.target}'
    
    st_time = time.time()
    BUFFER_SIZE = int(8192*640/10)
    csv.field_size_limit(sys.maxsize)
    
    train = work.load_tsv_dataset(f'{path}/train.tsv')
    valid = work.load_tsv_dataset(f'{path}/dev.tsv')
    test = work.load_tsv_dataset(f'{path}/test.tsv')
    
    wakati = MeCab.Tagger("-Owakati -b 5242880")

    for w in tqdm(train):
        content = w['content']
        w['wakati'] = ""
        if len(content) > BUFFER_SIZE:
            content = content[:BUFFER_SIZE]
        parsed = wakati.parse(content)
        try:
            tokens = parsed.split()
        except AttributeError as err:
            print("error:", err)
            print("length:", len(content))
            print("content:", parsed)
        w['wakati'] += " ".join(tokens) + " "

    for w in tqdm(valid):
        content = w['content']
        w['wakati'] = ""
        if len(content) > BUFFER_SIZE:
            content = content[:BUFFER_SIZE]
        parsed = wakati.parse(content)
        try:
            tokens = parsed.split()
        except AttributeError as err:
            print("error:", err)
            print("length:", len(content))
            print("content:", parsed)
        w['wakati'] += " ".join(tokens) + " "

    for w in tqdm(test):
        content = w['content']
        w['wakati'] = ""
        if len(content) > BUFFER_SIZE:
            content = content[:BUFFER_SIZE]
        parsed = wakati.parse(content)
        try:
            tokens = parsed.split()
        except AttributeError as err:
            print("error:", err)
            print("length:", len(content))
            print("content:", parsed)
        w['wakati'] += " ".join(tokens) + " "
    
    x_train = [w['wakati'] for w in train]
    x_valid = [w['wakati'] for w in valid]
    x_test  = [w['wakati'] for w in test ]

    y_train = [w['label'] for w in train]  
    y_valid = [w['label'] for w in valid]
    y_test  = [w['label'] for w in test ]

    del(train)
    del(valid)
    del(test)
    
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
    classifier.fit(x_train, y_train)
    traintime = time.strftime("%H:%M:%S", time.gmtime(time.time()-st_time))
    
    predicted = classifier.predict(x_valid)
    val_acc = accuracy_score(y_valid, predicted)
    
    predicted = classifier.predict(x_test)
    test_acc = accuracy_score(y_test, predicted)
    
    
    now = datetime.now()
    res_str = f'result-{now.strftime("%Y-%m-%d")}'
    save_path = f'./savepoint/lr/{args.dataset}/{args.target}'
    
    
    os.makedirs(save_path, exist_ok=True)
        
    with open(f'{save_path}/{res_str}.pkl','wb') as f:
        pickle.dump(classifier, f)
        
    with open(f'./savepoint/lr/{args.dataset}/{res_str}.jsonl', 'a+', encoding='utf-8') as file:
        str_time = now.strftime("%Y-%m-%d %H:%M:%S")
        val_result = {"label": args.target, "type": "dev", "date": str_time, "training time": traintime, "accuracy": float("{:.4f}".format(val_acc))}
        json.dump(val_result, file)
        file.write("\n")
        test_result = {"label": args.target, "type": "test", "date": str_time, "training time": traintime, "accuracy": float("{:.4f}".format(test_acc))}
        json.dump(test_result, file)
        file.write("\n")
        
if __name__ == "__main__":
    parser = ArgumentParser(description="Baseline LR")
    parser.add_argument('--dataset', type=str, default='full')
    parser.add_argument('--target', type=str, default='1')
    args = parser.parse_args()
    
    print(f'LR DATASET:{args.dataset} LABEL:{args.target}')
    main(args)
