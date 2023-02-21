import csv
import time
import json
from tqdm import tqdm
from os import walk

class Work:
    def __init__(self, i, t, c, l):
        self.title = t
        self.content = c
        self.labels = l
        self.id = i
    def __str__(self):
        return f"id: \"{self.id}\"\ntitle: \"{self.title}\"\ncontent: \"{self.content}\"\nlabels: {self.labels}\n"
    
def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

def load_csv(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as file:
        rows = csv.reader(file)
        for row in rows:
            label = {"id": row[0], "name": row[1]}
            data.append(label)
    return data

def add_str(datas):
    string = ""
    for data in datas:
        t_str = data['body']
        t_str = t_str.replace(u'\u3000', u'')
        string += t_str
    return string

def create_work(data, size):
    labels = load_csv("./dataset/label_list.csv")
    if data['labels'] != None:
        t_labels = create_label_vector(labels, data['labels'])
    else:
        t_labels = create_label_vector(labels, [""])
    if size > 0:
        w = Work(data['id'], data['metadata']['title'], add_str(data['content'])[:size], t_labels)
    else:
        w = Work(data['id'], data['metadata']['title'], add_str(data['content']), t_labels)   
    return w

def create_label_vector(total_labels, target_labels):
    return_label = []
    for i, label in enumerate(total_labels):
        for t_label in target_labels:
            if label['name'] == t_label:
                return_label.append(label['id'])
    return return_label

def read_cleaned_works(path, size):
    works = []
    st_time = time.time()
    filenames = next(walk(path),  (None, None, []))[2]
    for filename in filenames:   ### TEST ###
        datas = load_jsonl(f"{path}/{filename}")
        print(f"{filename} read lines: {len(datas)}")
        for data in tqdm(datas):
            w = create_work(data, size)
            work = {"id": w.id, "title": w.title, "content": w.content, "labels": w.labels, 'wakati':""}
            works.append(work)
    print(f"Time: {time.time()-st_time}")
    return works

def get_work_list_by_label(works, id):
    work_list = []
    for work in works:
        if (str(id) in work['labels']):
            label = 1
        else:
            label = 0
        work_list.append({'id': work['id'], 'title': work['title'], 'label': label, 'content': work['content']})
    return work_list

def load_tsv_dataset(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        rd = csv.reader(f, delimiter="\t")
        for row in rd:
            if row[0] == '01':
                label = 0
            elif row[0] == '10':
                label = 1
            rows.append({'label':label, 'content':row[1]})
    return rows

def load_tsv_label(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        rd = csv.reader(f, delimiter='\t')
        for row in rd:
            rows.append({'label':row[0], 'name':row[1:]})
    return rows

def load_tsv_morpheme(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd, None) # Skip headder
        for row in tqdm(rd):
            rows.append({'word': row[0], 'count':row[1]})
    return rows

def load_tsv_similarity(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd, None) # Skip headder
        for row in tqdm(rd):
            rows.append({'word': row[0], 'score':row[1]})
    return rows

def load_tsv_confidence(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        rd = csv.reader(f, delimiter='\t')
        for row in tqdm(rd):
            rows.append({'content': row[2], 'confidence': row[0]})
    return rows