import torch
import glob
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

category_lines = {}
all_categories = []

fasttext_data = []

test_set_data = []
test_set_labels = []
test_set = []

for filename in findFiles('./names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    test_set.append(lines[0]+ ' ' + '__label__' + category)
    test_set_data.append(lines[0])
    test_set_labels.append(category)
    category_lines[category] = lines[1:]
    for line in lines[1:]:
        fasttext_data.append(line + ' ' + '__label__' + category)

with open('./names/fasttext_data', 'w') as f:
    for s in fasttext_data:
        f.write(s + '\n')

n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
