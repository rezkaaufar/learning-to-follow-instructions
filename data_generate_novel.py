import sys
import itertools
import logging
import random
import copy
logging.basicConfig(level=logging.INFO)

show_progress= False
n_grammars = 5000
n_examples = 4
K = 3  # maximum size of k-factors
k = 3  # minumum size of k-factors
F = 5  # number of k-factors in a grammar
example_length = 20
alphabet_size = 6
alphabet ="".join(chr(i) for i in range(ord('a'), ord('a') + alphabet_size))
novel = False
novel_amount = 12

k_fact = list(itertools.chain(*(itertools.combinations_with_replacement(alphabet, i) for i in range(k,K+1))))
k_fact = list(map(lambda x:"".join(x), k_fact))

k_fact_novel = random.sample(k_fact, novel_amount)
k_fact = [x for x in k_fact if x not in k_fact_novel]

logging.debug("Set of possible k-factors: {}".format(k_fact))

def gen_grammar(novel=False):
    if not novel:
        g = []
        k_fact_pool = copy.copy(k_fact)
        while len(g) < F:
            if len(k_fact_pool) == 0:
                raise RuntimeError("No more k-factors to choose when generating "
                        "grammar {} with size {}".format(g, F))
            # choose a candidate k-factor
            kf = random.choice(k_fact_pool)
            # add it to the grammar
            g.append(kf)
            # remove from the k-factor pool those that are a superset of the chosen kf
            k_fact_pool = [f for f in k_fact_pool if kf not in f]
        return list(sorted(g))
    else:
        g = []
        k_fact_pool = copy.copy(k_fact_novel)
        while len(g) < F:
            if len(k_fact_pool) == 0:
                raise RuntimeError("No more k-factors to choose when generating "
                                   "grammar {} with size {}".format(g, F))
            # choose a candidate k-factor
            kf = random.choice(k_fact_pool)
            # add it to the grammar
            g.append(kf)
            # remove from the k-factor pool those that are a superset of the chosen kf
            k_fact_pool = [f for f in k_fact_pool if kf not in f]
        return list(sorted(g))

def belongs(g, s):
    for kf in g:
        if kf in s:
            return False
    return True
    
def gen_example(g, cls):
    # very inefficient solution for some grammars (those that contain very few
    # positive examples)
    done = False
    while not done:
        s = "".join(random.choices(alphabet, k=example_length))
        if belongs(g, s) == cls:
            return s

f = open("subreg_train_5000_nvlex.txt", "w")
ft = open("subreg_test_5000_nvlex.txt", "w")

train_composition = 0.8 * n_grammars
batch_example = 0
for i in range(n_grammars):
    if i < train_composition:
        g = gen_grammar()
    else:
        g = gen_grammar(True)
    g_str = "#".join(g)
    for j in range(n_examples):
        if show_progress:
            sys.stderr.write("\r{:02.2f}%".format(100*(i*n_examples+j)/n_grammars/n_examples))
        pos = gen_example(g, True)
        neg = gen_example(g, False)
        #if batch_example + (j+1) <= train_composition + batch_example :
        if i < train_composition:
            f.write("\t".join([g_str, pos, "1"]) + "\n")
            f.write("\t".join([g_str, neg, "0"]) + "\n")
        else:
            ft.write("\t".join([g_str, pos, "1"]) + "\n")
            ft.write("\t".join([g_str, neg, "0"]) + "\n")
    batch_example += n_examples

f.close()
ft.close()

