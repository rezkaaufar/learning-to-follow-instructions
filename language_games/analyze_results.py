#!/usr/bin/env python
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('data')
    ap.add_argument('-v', '--val', type=int, default=0)

    args = ap.parse_args()

    res = open(args.results).readlines()[1]
    res = eval(res)

    data = open(args.data).readlines()

    example_correct = res[2+args.val]#[args.start::2]
    chosen_model = res[4+args.val]#[args.start::2]
    assert len(data) == len(example_correct), "{}!={}".format(len(data), len(example_correct))
    assert len(data) == len(chosen_model), "{}!={}".format(len(data), len(chosen_model))
    for r,m, s in zip(example_correct, chosen_model, data):
        print("{}\t{}\t{}".format(r, m, s.rstrip('\n')))



main()
