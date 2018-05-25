
#!/usr/bin/env python
import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('data', nargs='?')
    ap.add_argument('-v', '--val', type=int, default=0)

    args = ap.parse_args()

    res = open(args.results).readlines()[1]
    res = eval(res)

    if not args.data:
        x = os.path.basename(args.results)
        x = x.replace('e-0', 'e_0')
        kw = dict(fs.split("_", 1) for fs in x.split("-"))
        args.data = 'dataset/sida wang\'s/txt/' + kw['data']
    data = open(args.data).readlines()

    example_correct = res[2+args.val]#[args.start::2]
    chosen_model = res[4+args.val]#[args.start::2]
    assert len(data) == len(example_correct), "{}!={}".format(len(data), len(example_correct))
    assert len(data) == len(chosen_model), "{}!={}".format(len(data), len(chosen_model))
    for r,m, s in zip(example_correct, chosen_model, data):
        print("{}\t{}\t{}".format(r, m, s.rstrip('\n')))



main()
