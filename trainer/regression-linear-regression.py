import os
import csv
import sys
import json
import time
import pickle
import workFactory as wf

from argparse import ArgumentParser
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def main(args):
    target = args.target
    workPath = f'./tsv/nearest-k-confidence-BERT/n-100/o-256/{target}'
    savePath = f'./savepoint/regression/LinearRegression/{args.target}'
    
    trainWorks = wf.load_tsv_confidence(f'{workPath}/train.tsv')
    x_train = [w['content'] for w in trainWorks]
    y_train = [w['confidence'] for w in trainWorks]
    
    devWorks = wf.load_tsv_confidence(f'{workPath}/dev.tsv')
    x_dev = [w['content'] for w in devWorks]
    y_dev = [w['confidence'] for w in devWorks]
    
    testWorks = wf.load_tsv_confidence(f'{workPath}/test.tsv')
    x_test = [w['content'] for w in testWorks]
    y_test = [w['confidence'] for w in testWorks]
    
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', LinearRegression())
    ])
    
    st_time = time.time()
    classifier.fit(x_train, y_train)
    traintime = time.strftime("%H:%M:%S", time.gmtime(time.time()-st_time))
    
    pred_dev = classifier.predict(x_dev)
    r2_dev = r2_score(y_dev, pred_dev)
    mae_dev = mean_absolute_error(y_dev, pred_dev)
    
    print(f'DEV:\n\tR2: {r2_dev}\n\tMAE: {mae_dev}')
    
    pred_test = classifier.predict(x_test)
    r2_test = r2_score(y_test, pred_test)
    mae_test = mean_absolute_error(y_test, pred_test)
    
    print(f'TEST:\n\tR2: {r2_test}\n\tMAE: {mae_test}')
    
    os.makedirs(savePath, exist_ok=True)
    with open(f'{savePath}/lr-1st.pkl', 'wb') as f:
        pickle.dump(classifier, f)
        
    now = datetime.now()
    res_str = f'result-{now.strftime("%Y-%m-%d")}'
    
    with open(f'{savePath}/../{res_str}.jsonl', 'a', encoding='utf-8') as file:
        str_time = now.strftime("%Y-%m-%d %H:%M:%S")
        dev_result = {
            'target': target, 
            'type': 'dev', 
            'date': str_time, 
            'train_time': traintime, 
            'r2': float("{:.4f}".format(r2_dev)), 
            'mae': float("{:.4f}".format(mae_dev))
        }
        json.dump(dev_result, file)
        file.write('\n')
        test_result = {
            'target': target, 
            'type': 'test', 
            'date': str_time, 
            'train_time': traintime, 
            'r2': float("{:.4f}".format(r2_test)), 
            'mae': float("{:.4f}".format(mae_test))
        }
        json.dump(test_result, file)
        file.write('\n')

if __name__ == "__main__":
    parser = ArgumentParser(description='Regression LinearRegression')
    parser.add_argument('--target', type=str, default='42')
    args = parser.parse_args()
    
    print(f'LinearRegression\n\tLabel:{args.target}')
    main(args)