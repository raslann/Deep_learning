

def preprocess_vocab(filename):
    '''
    Preprocess the text and get the vocabulary from it.
    The vocabulary is stored in filename+'.vcb', where every word is written
    line by line.
    '''
    vocab = set()
    vocab_filename = filename + '.vcb'
    file_ = open(filename, 'r')

    while True:
        line = file_.readline()
        if line == '':
            break

        words = line.strip().split(' ')
        vocab |= set(words)

    vocab_list = list(vocab)

    vocab_file = open(vocab_filename, 'w')
    vocab_file.writelines([w + '\n' for w in vocab_list])

    vocab_file.close()
    file_.close()


def preprocess(filename, vocab_filename=None):
    '''
    Generate a tokenized file where each word is replaced by an index
    in the vocabulary file, and generate an index file where every
    entry is the byte offset of the tokenized file.

    Necessary for training large corpus.
    '''
    if vocab_filename is None:
        vocab_filename = filename + '.vcb'
    token_filename = filename + '.tok'
    index_filename = filename + '.idx'
    file_ = open(filename, 'r')

    vocab_dict = {}

    vocab_file = open(vocab_filename, 'r')
    word_index = 0
    while True:
        word = vocab_file.readline().strip()
        if word == '':
            break
        vocab_dict[word] = word_index
        word_index += 1
    vocab_file.close()

    token_file = open(token_filename, 'w')
    index_file = open(index_filename, 'w')
    index = 0

    while True:
        line = file_.readline()
        if line == '':
            break

        words = line.strip().split(' ')
        token_line = ' '.join([str(vocab_dict[w]) for w in words]) + '\n'
        index_file.write('%d\n' % index)
        index += len(token_line)
        token_file.write(token_line)

    index_file.close()
    token_file.close()
    file_.close()
