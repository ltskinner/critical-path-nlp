

import json

dpath = "C:\\Users\\Angus\\data\\SQuAD_2.0\\small-train-2.0.json"
with open(dpath) as f:
    data = json.load(f)

qas = {}

for sample in data["data"]:
    for paras in sample["paragraphs"]:
        for qa in paras['qas']:
            qid = qa['id']
            question = qa['question'] 
            qas[qid] = {
                'q': question
            }


apath = "C:\\Users\\Angus\\models\\uncased_L-12_H-768_A-12\\smol\\predictions.json"

with open(apath) as f:
    answers = json.load(f)

for tag in answers.keys():
    qas[tag]['a'] = answers[tag]


for qid in qas.keys():
    print("--------------------------------------------------------------")
    print(qas[qid]['q'])
    print(qas[qid]['a'])
