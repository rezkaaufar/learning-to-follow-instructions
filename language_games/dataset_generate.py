from random import randint
import random
import copy
import itertools

# we assume that there are 6 tiles and maximum 3 blocks
color_blocks = ["0", "1", "2", "3"] # cyan, brown, red, orange
color_strings = ["cyan", "brown", "red", "orange"]
operation = ["remove", "add"]
positional_info = ["1st", "2nd", "3rd", "4th", "5th", "6th", "even", "odd", "leftmost", "rightmost", "every"]
num_tiles, max_block, num_color = 6, 3, 3

### for blocks novel ###
alphabet = ["0","1","2","3","X"]
novel_amount = 8
blocks = [''.join(p) for p in itertools.product(alphabet, repeat=3)]
blocks_n = []
for bl in blocks:
  if 'X' in bl[1] and bl[2] is not 'X':
    blocks_n.append(bl)
  elif 'X' in bl[0] and bl[1] is not 'X':
    blocks_n.append(bl)
# length : 85
blocks_fin = [x for x in blocks if x not in blocks_n]
#blocks = list(itertools.chain(*(itertools.permutations(alphabet, i) for i in range(3,3+1))))
#blocks = list(map(lambda x:"".join(x), blocks))
blocks_test = random.sample(blocks_fin, novel_amount)
blocks_mt = [x for x in blocks_fin if x not in blocks_test]
blocks_valid = random.sample(blocks_mt, novel_amount)
blocks = [x for x in blocks_mt if x not in blocks_valid]

### for utterance novel ###
def add_info(string):
  ins = string.split(" ")
  ins.insert(3, "tile")
  if ins[0] == "remove":
    ins.insert(2, "at")
  elif ins[0] == "add":
    ins.insert(2, "to")
  return " ".join(ins)
command_list = [["remove", "add"], ["cyan", "brown", "red", "orange"],
                ["1st", "2nd", "3rd", "4th", "5th", "6th", "even", "odd", "leftmost", "rightmost", "every"]]
# command_list = [["remove", "add"], ["cyan", "brown", "red", "orange"],
#                 ["1st", "2nd", "3rd", "4th", "5th", "6th", "even", "odd", "leftmost", "rightmost", "every"]]
# length : 88
novel_amount = 5
command = [add_info(" ".join(p)) for p in itertools.product(*command_list)]
r_command = command[:44]
a_command = command[44:]
rmv_test = random.sample(r_command, novel_amount)
add_test = random.sample(a_command, novel_amount)
add_command_tmp = [x for x in a_command if x not in add_test]
rmv_command_tmp = [x for x in r_command if x not in rmv_test]
rmv_valid = random.sample(rmv_command_tmp, novel_amount)
add_valid = random.sample(add_command_tmp, novel_amount)
add_command = [x for x in add_command_tmp if x not in add_valid]
rmv_command = [x for x in rmv_command_tmp if x not in rmv_valid]
# command_fin = [add_info(val) for pair in zip(add_command, rmv_command) for val in pair]
# command_fin_test = [add_info(val) for pair in zip(add_test, rmv_test) for val in pair]

def generate_train_data_novel(num_tiles, max_block, num_color, opr, blocks_b, utter_b,
                              added_set, blocks, command_set):
  match = False
  ## user command ##
  if utter_b:
    command = random.choice(command_set)
  else:
    clr = random.choice(color_strings)
    pos = random.choice(positional_info)
    if opr == "remove":
      info = " at "
    else:
      info = " to "
    command = opr + " " + clr + info + pos + " tile"

  ## initial state ##
  already_added = True
  if blocks_b:
    while already_added:
      block_config = []
      block_cfg = random.sample(blocks, num_tiles)
      for i in range(num_tiles):  # loop through all the tiles
        x = list(block_cfg[i])
        x = list(filter(lambda a: a != 'X', x))
        block_config.append(x)
      if str(block_config) not in added_set:
        already_added = False
  else:
    while already_added:
      block_config = []
      for i in range(num_tiles):  # loop through all the tiles
        num_blocks = randint(0, max_block)
        if num_blocks == 0:
          block_config.append([])
        else:
          blocks = []
          for j in range(num_blocks):
            block_color_idx = randint(0, num_color)
            blocks.append(color_blocks[block_color_idx])
          block_config.append(blocks)
      if str(block_config) not in added_set:
        already_added = False

  ## generate next state ##
  next_block_config = copy.deepcopy(block_config)
  tile_indexs = [-1]
  if "1" in command or "leftmost" in command:
    tile_indexs = [0]
  elif "2" in command:
    tile_indexs = [1]
  elif "3" in command:
    tile_indexs = [2]
  elif "4" in command:
    tile_indexs = [3]
  elif "5" in command:
    tile_indexs = [4]
  elif "6" in command or "rightmost" in command:
    tile_indexs = [5]
  elif "even" in command:
    tile_indexs = [1, 3, 5]
  elif "odd" in command:
    tile_indexs = [0, 2, 4]
  elif "every" in command:
    tile_indexs = [0, 1, 2, 3, 4, 5]

  for tile_index in tile_indexs:
    if "remove" in command:
      if len(next_block_config[tile_index]) != 0:
        if "cyan" in command:
          if next_block_config[tile_index][-1] == "0":
            next_block_config[tile_index].pop()
            match = True
        elif "brown" in command:
          if next_block_config[tile_index][-1] == "1":
            next_block_config[tile_index].pop()
            match = True
        elif "red" in command:
          if next_block_config[tile_index][-1] == "2":
            next_block_config[tile_index].pop()
            match = True
        elif "orange" in command:
          if next_block_config[tile_index][-1] == "3":
            next_block_config[tile_index].pop()
            match = True
    elif "add" in command and len(next_block_config[tile_index]) < 3:
      if "cyan" in command:
        next_block_config[tile_index].append("0")
        match = True
      elif "brown" in command:
        next_block_config[tile_index].append("1")
        match = True
      elif "red" in command:
        next_block_config[tile_index].append("2")
        match = True
      elif "orange" in command:
        next_block_config[tile_index].append("3")
        match = True
  if match:
    added_set.add(str(block_config))

  # change block_config
  for i, el in enumerate(block_config):
    rs = 3 - len(el)
    for j in range(rs):
      block_config[i].append("X")
  block_config = list(itertools.chain(*block_config))
  for i in range(1, 6):
    block_config.insert(i * 3 + i - 1, '#')

  for i, el in enumerate(next_block_config):
    rs = 3 - len(el)
    for j in range(rs):
      next_block_config[i].append("X")
  next_block_config = list(itertools.chain(*next_block_config))
  for i in range(1, 6):
    next_block_config.insert(i * 3 + i - 1, '#')

  return block_config, command, next_block_config, match, added_set

total_train = 42000
total_test = 4000
total_valid = 4000
total = total_train + total_test + total_valid

mode = "utter"
blocks_b = False
utter_b = True

f = open("./lang_games_data_artificial_train_nvl_" + mode + "_" + str(total) + ".txt", "w")
ft = open("./lang_games_data_artificial_test_nvl_" + mode + "_" + str(total) + ".txt", "w")
fv = open("./lang_games_data_artificial_valid_nvl_" + mode + "_" + str(total) + ".txt", "w")
added_set = set()

rc_train, rc_valid, rc_test = rmv_command, rmv_valid, rmv_test
ac_train, ac_valid, ac_test = add_command, add_valid, add_test


for i in range(total_train):
  opr = "add"
  if i < 0.5 * total_train:
    opr = "remove"
  match = False

  while not match:
    if i < 0.5 * total_train:
        block_config, command, next_block_config, match, added_set = generate_train_data_novel(
            num_tiles, max_block, num_color, opr, blocks_b, utter_b, added_set, blocks, rc_train)
        if match:
            f.write(" ".join(bc for bc in block_config)
                     + "\t" + command + "\t" + " ".join(bc for bc in next_block_config) + "\n")
    else:
        block_config, command, next_block_config, match, added_set = generate_train_data_novel(
            num_tiles, max_block, num_color, opr, blocks_b, utter_b, added_set, blocks, ac_train)
        if match:
            f.write(" ".join(bc for bc in block_config)
                    + "\t" + command + "\t" + " ".join(bc for bc in next_block_config) + "\n")
f.close()

for i in range(total_test):
  opr = "add"
  if i < 0.5 * total_test:
      opr = "remove"
  match = False

  while not match:
    if i < 0.5 * total_test:
        block_config, command, next_block_config, match, added_set = generate_train_data_novel(
            num_tiles, max_block, num_color, opr, blocks_b, utter_b, added_set, blocks_test, rc_test)
        if match:
            ft.write(" ".join(bc for bc in block_config)
                     + "\t" + command + "\t" + " ".join(bc for bc in next_block_config) + "\n")
    else:
        block_config, command, next_block_config, match, added_set = generate_train_data_novel(
            num_tiles, max_block, num_color, opr, blocks_b, utter_b, added_set, blocks_test, ac_test)
        if match:
            ft.write(" ".join(bc for bc in block_config)
                    + "\t" + command + "\t" + " ".join(bc for bc in next_block_config) + "\n")
ft.close()

for i in range(total_valid):
  opr = "add"
  if i < 0.5 * total_valid:
      opr = "remove"
  match = False

  while not match:
    if i < 0.5 * total_test:
        block_config, command, next_block_config, match, added_set = generate_train_data_novel(
            num_tiles, max_block, num_color, opr, blocks_b, utter_b, added_set, blocks_valid, rc_valid)
        if match:
            fv.write(" ".join(bc for bc in block_config)
                     + "\t" + command + "\t" + " ".join(bc for bc in next_block_config) + "\n")
    else:
        block_config, command, next_block_config, match, added_set = generate_train_data_novel(
            num_tiles, max_block, num_color, opr, blocks_b, utter_b, added_set, blocks_valid, ac_valid)
        if match:
            fv.write(" ".join(bc for bc in block_config)
                    + "\t" + command + "\t" + " ".join(bc for bc in next_block_config) + "\n")
fv.close()