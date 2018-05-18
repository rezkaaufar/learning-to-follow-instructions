#!/usr/bin/env python
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json_files', nargs='+')
    args = ap.parse_args()

    for fn in args.json_files:
        df = pd.read_json(fn)
        last_iter = df.iloc[-1]
        print (fn, last_iter['data/test_accuracy'], last_iter['data/val_accuracy'])

main()
