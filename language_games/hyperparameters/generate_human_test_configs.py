#!/usr/bin/env python
import pandas as pd
import os.path
import glob
import json

df = pd.read_json('hyperparameters/human_best_hps.json')
val_data = ["AZGBKAM5JUV5A", "A1HKYY6XI2OHO1", "ADJ9I7ZBFYFH7"]
data = []
for f in glob.glob('dataset/sida wang\'s/txt/*.txt'):
    f = os.path.basename(f)
    f = os.path.splitext(f)[0]
    if f not in val_data:
        data.append("sida wang's/txt/" + f)

for i,r in df.iterrows():
    params = {}
    ks = ['k', 'lamb', 'learner', 'lr', 'optim', 'regularize', 'steps', 'unfreezed']
    for k in ks:
        params[k] = [r[k]]
    params['output'] = ['results_human_test']
    params['data'] = data
    with open('hyperparameters/human_test_{}.json'.format(r['unfreezed']), 'w') as fout:
        json.dump(params, fout)

