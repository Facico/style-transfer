import re
import os
import fasttext
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default='yelp_log', help='input file')
# parser.add_argument("--file", type=str, default='siamese', help='input file')
parser.add_argument("--out", type=str, default="out", help="output file")
parser.add_argument("--it", type=int, default=None)
parser.add_argument("--change", type=int, default=None)
parser.add_argument("--pos_c", type=int, default=0)
parser.add_argument("--neg_c", type=int, default=0)
parser.add_argument("--early", action='store_true')
parser.add_argument("--partial", action='store_true')

args = parser.parse_args()

# if args.it == None and args.change == None:
#     raise ValueError('it and change are all None')

# if args.it != None and args.change != None:
#     raise ValueError('it and change are all not None')

def get_score_label(text, Dis_model, goal):
    result = Dis_model.predict(text.strip('\n'))
    dis_score = result[1][0]
    label_now = int(result[0][0].split('__')[-1])
    if (label_now != goal):
        dis_score = 1 - dis_score
    label_now = int(result[0][0].split('__')[-1])
    return dis_score, label_now


file = args.file
sentences = []
with open(file, 'r') as fr:
    sentences = fr.readlines()
# fr = open(file, 'r')

iteration = args.it
change = args.change
sent_id = -1
uv = -1
text021 = []
text120 = []
res = {}

for i in sentences:
    o = re.finditer('Sentence Pair.', i)
    first = -1
    for j in o:
        first = 1
        sent_id += 1
        break
    if(first == 1):
        continue

    o = re.finditer('iteration.', i)
    first = -1
    for j in o:
        first = j.start()
        break
    if(first != -1):
        first = first + len('iteration.')
        sent_split = i[first:].split(',')
        it_id = int(sent_split[0])
        dict_name = 'iteration.{:d}'.format(it_id)
        if dict_name not in res.keys():
            res[dict_name] = {}
        res[dict_name][sent_id] = i[first+len(str(it_id))+2:]

    o = re.finditer('Change.', i)
    first = -1
    for j in o:
        first = j.start()
        break
    if(first != -1):
        first = first + len('Change.')
        sent_split = i[first:].split(',')
        it_id = int(sent_split[0])
        dict_name = 'Change.{:d}'.format(it_id)
        if dict_name not in res.keys():
            res[dict_name] = {}
        res[dict_name][sent_id] = i[first+len(str(it_id))+2:]
    
    o = re.finditer('ATTACK.', i)
    first = -1
    for j in o:
        first = j.start()
        break
    if first != -1:
        first = first + len('ATTACK.')
        sent_split = i[first:]
        if 'attack' not in res.keys():
            res['attack'] = {}
        res['attack'][sent_id] = sent_split

print('-'*30)

if args.early:
    if os.path.exists(args.out):
        os.remove(args.out)
    limit = 50 if args.partial else 1000
    for i in range(limit):
        if i in res['attack']:
            sent = res['attack'][i]
        else:
            sent = res['iteration.0'][i]
        with open(args.out, 'a') as fout:
            fout.write(sent)
    exit(0)


if args.it != None:
    if os.path.exists(args.out):
        os.remove(args.out)
    limit = 50 if args.partial else 1000
    for i in range(limit):
        sent = res['iteration.{:d}'.format(args.it)][i]
        with open(args.out, 'a') as fout:
            fout.write(sent)

else:
    if os.path.exists(args.out):
        os.remove(args.out)
    limit = 50 if args.partial else 1000
    for i in range(limit):
        change = args.change if args.change != None else args.pos_c if i < 500 else args.neg_c
        sent = None
        for cur_change in range(change, -1, -1):
            res_name = 'Change.{:d}'.format(cur_change)
            if res_name in res.keys() and i in res[res_name]:
                sent = res[res_name][i]
                break
        if sent == None:
            sent = res['iteration.0'][i]

        with open(args.out, 'a') as fout:
            fout.write(sent)

print('Done!')