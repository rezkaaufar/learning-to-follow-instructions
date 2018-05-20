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
    ap.add_argument('-v', '--val')
    args = ap.parse_args()

    df = None
    for i,l in enumerate(fileinput.input(args.results)):
        l = l.rstrip('\n')
        # hacks
        l = l.replace('e-0', 'e_0')
        if i%2 == 0:
            name = l
            try:
                features = dict(fs.split("_", 1) for fs in l.split("-"))
            except ValueError:
                print(l)
                raise
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
            df.loc[idx,'greedy'] = float(res[0]) #if args.val <2 else max(float(res[0]),float(res[1]))
            df.loc[idx,'1-out'] = float(res[1]) #if args.val <2 else max(float(res[0]),float(res[1]))
            #df.loc[idx,'accuracy']= float(res[args.val]) if args.val <2 else max(float(res[0]),float(res[1]))
            if len(res) > 6:
                df.loc[idx,'back_accuracy']= float((res[-2][-1]))
    if df is None:
        print("No input")
        exit(1)
    val_vars = ['greedy', '1-out']
    df = pd.melt(df, id_vars=list(c for c in df.columns if c not in val_vars),
            value_vars=val_vars, value_name='accuracy', var_name='model-select')

    df.reset_index(level=0, inplace=True)
    #df = df.reset_index(drop=True)
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
    if args.val:
        df = df[df['model-select'] == args.val]
    best_res_all = df.loc[df.groupby(group_by)[measure].idxmax().drop_duplicates()]
    best_res = best_res_all.groupby(group_by).first().reset_index()
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    print(best_res_all.sort_values(group_by))


main()
