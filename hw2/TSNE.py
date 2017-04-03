import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from rnn import LanguageModel
from util import variable
import numpy as np


def main():
    model = torch.load('rnn.pt')
    embeddings = model.x
    vocab_file = open('ptb/train.txt.vcb', "r")
    vocabulary = [w.strip() for w in vocab_file.readlines()]
    vocab_size = len(vocabulary)
    vocab_file.close()

    words = variable(torch.LongTensor(range(vocab_size)).cuda())
    wv = embeddings(words).data.cpu()numpy()

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:1000, :])

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
    plt.savefig('wordsssss')


if __name__ == '__main__':
    main()



