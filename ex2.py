
import sys
import math

development_file = sys.argv[1]
test_file = sys.argv[2]
input_word = sys.argv[3]
output_file = sys.argv[4]

vocab_size = 3e5

print '#Output1\t' + development_file
print '#Output2\t' + test_file
print '#Output3\t' + input_word
print '#Output4\t' + output_file
print '#Output5\t' + str(int(vocab_size))
print '#Output6\t' + str(1 / vocab_size)


d_f = open(development_file, 'r')
develop_set = []
for line in d_f:
    line = line.strip()
    if line[0:1] == '<' and line[-1:] == '>':
        continue
    if line == '':
        continue
    events = line.split(' ')
    for e in events:
        develop_set.append(e)
d_f.close()

print '#Output7\t' + str(len(develop_set))

split_index = int(round(len(develop_set) * 0.9))
train_set = develop_set[0:split_index]
validate_set = develop_set[split_index:]

print '#Output8\t' + str(len(validate_set))
print '#Output9\t' + str(len(train_set))

train_word_dict = {}
validate_word_dict = {}

for word in train_set:
    if word not in train_word_dict:
        train_word_dict[word] = 0
    train_word_dict[word] += 1

for word in validate_set:
    if word not in validate_word_dict:
        validate_word_dict[word] = 0
    validate_word_dict[word] += 1

print '#Output10\t' + str(len(train_word_dict))
print '#Output11\t' + str(train_word_dict[input_word])

S_size = len(train_set)
X_size = vocab_size

p_input_word = train_word_dict[input_word] / float(S_size)
print '#Output12\t' + str(p_input_word)

p_not_seen = 0
print '#Output13\t' + str(p_not_seen)


def calculate_probability_Lidstone(word, lamda):
    word_count = 0
    if word in train_word_dict:
        word_count = train_word_dict[word]
    return (word_count + lamda) / (float(S_size) + lamda * X_size)


p_input_word = calculate_probability_Lidstone(input_word, 0.1)
print '#Output14\t' + str(p_input_word)

p_not_seen = calculate_probability_Lidstone('unseen-word', 0.1)
print '#Output15\t' + str(p_not_seen)


def calculate_perplexity(lamda):
    log_sum = 0
    for word in validate_set:
        p = calculate_probability_Lidstone(word, lamda)
        log_sum += math.log(p, 2)
    log_sum /= len(validate_set)
    return 2 ** (- log_sum)


print '#Output16\t' + str(calculate_perplexity(0.01))
print '#Output17\t' + str(calculate_perplexity(0.10))
print '#Output18\t' + str(calculate_perplexity(1.00))

best_lamda = 0.00
lowest_perplexity = float('inf')

for i in range(1, 201, 1):
    curr_lamda = i / float(100)
    curr_perplexity = calculate_perplexity(curr_lamda)
    if curr_perplexity < lowest_perplexity:
        best_lamda = curr_lamda
        lowest_perplexity = curr_perplexity

print '#Output19\t' + str(best_lamda)
print '#Output20\t' + str(lowest_perplexity)

halving_index = len(develop_set) / 2
train_set = develop_set[0:halving_index]
held_out_set = develop_set[halving_index:]

print '#Output21\t' + str(len(train_set))
print '#Output22\t' + str(len(held_out_set))

train_word_dict = {}
held_out_word_dict = {}

for word in train_set:
    if word not in train_word_dict:
        train_word_dict[word] = 0
    train_word_dict[word] += 1

for word in held_out_set:
    if word not in held_out_word_dict:
        held_out_word_dict[word] = 0
    held_out_word_dict[word] += 1

N_dict = {}
T_dict = {}

for word in train_word_dict:
    r = train_word_dict[word]
    r_str = str(r)
    if r_str not in N_dict:
        N_dict[r_str] = 0
    if r_str not in T_dict:
        T_dict[r_str] = 0
    N_dict[r_str] += 1
    T_dict[r_str] += held_out_word_dict.get(word, 0)

# calculating N[0]
N_0 = vocab_size
for r in N_dict:
    N_0 -= N_dict[r]
N_dict['0'] = N_0

# calculating T[0]
T_dict['0'] = 0
for word in held_out_word_dict:
    if word not in train_word_dict:
        T_dict['0'] += held_out_word_dict[word]


def calculate_probability_Heldout(word):
    r_str = str(train_word_dict.get(word, 0))
    return T_dict[r_str] / float(len(held_out_set) * N_dict[r_str])


p_input_word = calculate_probability_Heldout(input_word)
print '#Output23\t' + str(p_input_word)

p_not_seen = calculate_probability_Heldout('unseen-word')
print '#Output24\t' + str(p_not_seen)