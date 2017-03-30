import torch
from torch.autograd import Variable
from rnn import LanguageModel
from sklearn.manifold import TSNE
import numpy as np
from util import variable
import matplotlib.pyplot as Plot
import numpy as Math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



loaded_model = torch.load('rnn.pt')
embeddings = loaded_model.x

vocab_file = open('ptb/train.txt.vcb', "r")
vocab = [w.strip() for w in vocab_file.readlines()]
vocab_size = len(vocab)
vocab_file.close()

words = variable(torch.LongTensor(range(vocab_size)))
embed = embeddings(words).data.numpy()
#embed = embed[:5] #to_delete

#tsne = TSNE(n_components=2)
#reduced_matrix = tsne.fit_transform(embed)




plt.rcParams["figure.figsize"] = (18, 10)

def plot_words(*words, lines=False):
    # pca = PCA(n_components=50)
    # xys = pca.fit_transform(embed)
    tsne = TSNE(n_components=2)
    xys = tsne.fit_transform(embed)

    if lines:
        for i in range(0, len(words), 2):
            plt.plot(xys[i:i+2, 0], xys[i:i+2, 1])
    else:
        plt.scatter(*xys.T)

    for word, xy in zip(words, xys):
        plt.annotate(word, xy, fontsize=20)

    return pca


plot_words('stream', 'euro', 'baseball', 'mountain', 'computer', 'lake', 'yen',
           'monkey', 'dog', 'basketball', 'cat', 'river', 'piano')



# Plot.savefig("glove_2000.png");


