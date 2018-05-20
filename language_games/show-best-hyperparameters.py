#!/usr/bin/env python
import argparse
import fileinput
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='*')
    ap.add_argument('-n', '--add-names', action='store_true', default=False)
    ap.add_argument('-u', '--unfreezed')
    ap.add_argument('-l', '--lamb')
    ap.add_argument('-v', '--val', type=int, default=0)
    args = ap.parse_args()

    df = None
    for i,l in enumerate(fileinput.input(args.results)):
        l = l.rstrip('\n')
        # hacks
        l = l.replace('e-0', 'e_0')
        if i%2 == 0:
            name = l
            features = dict(fs.split("_", 1) for fs in l.split("-"))
            if df is None:
                df = pd.DataFrame(columns=list(features.keys()) + ['accuracy',
                    'back_accuracy', 'name'])
        else:
            res = eval(l)
            idx = (i-1)/2
            for k,v in features.items():
                df.loc[idx, k] = v
            if args.add_names:
                df.loc[idx,'name'] = name
            df.loc[idx,'accuracy']= float(res[args.val]) if args.val <2 else max(float(res[0]),float(res[1]))
            if len(res) > 6:
                df.loc[idx,'back_accuracy']= float((res[-2][-1]))
    if df is None:
        print("No input")
        exit(1)
    group_by = ['data', 'lr']
    measure='accuracy'
    dtypes = {'data': 'str', 'steps': 'int', 'accuracy': 'float',
            'back_accuracy': 'float'}
    for k,v in dtypes.items():
        df[k] = df[k].astype(v)
    if args.unfreezed:
        df = df[df['unfreezed'] == args.unfreezed]
    if args.lamb:
        df = df[df['lamb'] == args.lamb]
    best_res_all = df.loc[df.groupby(group_by)[measure].idxmax()]
    best_res = best_res_all.groupby(group_by).first().reset_index()
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    print(best_res)


main()
