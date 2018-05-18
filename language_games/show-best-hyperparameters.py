#!/usr/bin/env python
import argparse
import fileinput
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='*')
    args = ap.parse_args()

    df = None
    for i,l in enumerate(fileinput.input(args.results)):
        # hacks
        l = l.replace('e-0', 'e_0')
        if i%2 == 0:
            features = dict(fs.split("_", 1) for fs in l.split("-"))
            if df is None:
                df = pd.DataFrame(columns=list(features.keys()) + ['accuracy'])
        else:
            res = eval(l)
            idx = (i-1)/2
            for k,v in features.items():
                df.loc[idx, k] = v
            df.loc[idx,'accuracy']= float(res[0])
    group_by = ['data', 'steps']
    measure='accuracy'
    dtypes = {'data': 'str', 'steps': 'int'}
    for k,v in dtypes.items():
        df[k] = df[k].astype(v)
    df[measure] = df[measure].astype('float')
    best_res_all = df.loc[df.groupby(group_by)[measure].idxmax()]
    best_res = best_res_all.groupby(group_by).first().reset_index()
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(best_res)


main()
