import sys
import itertools
import logging
import random
import copy
logging.basicConfig(level=logging.INFO)

show_progress= False
n_grammars = 5
n_examples = 4000
K = 3  # maximum size of k-factors
k = 3  # minumum size of k-factors
F = 5  # number of k-factors in a grammar
example_length = 20
alphabet_size = 6
alphabet ="".join(chr(i) for i in range(ord('a'), ord('a') + alphabet_size))

k_fact= list(itertools.chain(*(itertools.combinations_with_replacement(alphabet, i) for i in range(k,K+1))))
k_fact = list(map(lambda x:"".join(x), k_fact))
logging.debug("Set of possible k-factors: {}".format(k_fact))

def gen_grammar():
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

f = open("dataset/overlap/subreg_train_" + str(n_grammars) +  ".txt", "w")
ft = open("dataset/overlap/subreg_test_" + str(n_grammars) +  ".txt", "w")
fv = open("dataset/overlap/subreg_valid_" + str(n_grammars) +  ".txt", "w") # "_" + str(n_examples) +

train_composition = 0.8 * n_examples * n_grammars
rest_composition = 0.1 * n_examples * n_grammars
batch_example = 0

# generate training
grammar_pool = []
for i in range(n_grammars):
    g = gen_grammar()
    g_str = "#".join(g)
    grammar_pool.append((g_str, g))
    for j in range(n_examples):
        if show_progress:
            sys.stderr.write("\r{:02.2f}%".format(100*(i*n_examples+j)/n_grammars/n_examples))
        pos = gen_example(g, True)
        neg = gen_example(g, False)
        if batch_example + (j+1) <= train_composition:
            f.write("\t".join([g_str, pos, "1"]) + "\n")
            f.write("\t".join([g_str, neg, "0"]) + "\n")
        # elif batch_example + (j+1) <= train_composition + rest_composition + batch_example:
        #     fv.write("\t".join([g_str, pos, "1"]) + "\n")
        #     fv.write("\t".join([g_str, neg, "0"]) + "\n")
    batch_example += n_examples
    if batch_example + 1 > train_composition:
        break

for i in range(int(rest_composition)):
    gs = random.choice(grammar_pool)
    pos = gen_example(gs[1], True)
    neg = gen_example(gs[1], False)
    fv.write("\t".join([gs[0], pos, "1"]) + "\n")
    fv.write("\t".join([gs[0], neg, "0"]) + "\n")

for i in range(int(rest_composition)):
    gs = random.choice(grammar_pool)
    pos = gen_example(gs[1], True)
    neg = gen_example(gs[1], False)
    ft.write("\t".join([gs[0], pos, "1"]) + "\n")
    ft.write("\t".join([gs[0], neg, "0"]) + "\n")


f.close()
ft.close()
fv.close()
