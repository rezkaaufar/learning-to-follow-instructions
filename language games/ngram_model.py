from collections import *

def train_char_lm(fname, order=4):
    with open(fname, 'r') as f:
      lm = defaultdict(Counter)
      #pad = "~" * order
      for line in f:
        ln = line.replace("\t"," ").replace("\n","")
        chars = ln.split(" ")
        for i in range(len(chars)-order):
          history, char = " ".join(chars[i:i+order]), chars[i+order]
          lm[history][char]+=1
      def normalize(counter):
          s = float(sum(counter.values()))
          return [(c,cnt/s) for c,cnt in counter.items()]
      outlm = {hist:normalize(chars) for hist, chars in lm.items()}
      return outlm

which_data = "utter_blocks"
lm = train_char_lm("./dataset/lang_games_data_artificial_test_nvl_"
                                                    + which_data + "_50000.txt")
for key, value in lm.items():
  print(key, value)