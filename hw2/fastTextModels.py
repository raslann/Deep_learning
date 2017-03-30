from fastTextPreprocess import *
from model import *
from train import *
from predict import predict

import numpy as np
import fasttext

'''
You can install the dependency via: pip install fasttext.

To load the cbow model: model = fasttext.load('cbow.bin').

To load the skip gram model: model = fasttext.load('skip_gram.bin').

To get the number of words: len(model.words).  #9859

To get the size of each word vector: model.dim. #100
'''

# Skip Gram word-vectors
model_skipgram = fasttext.skipgram('./ptb/train.txt', 'skip_gram', epoch=100000)

# Continuous bag-of-words word-vectors
model_cbow = fasttext.cbow('./ptb/train.txt', 'cbow', epoch=100000)

start = time.time()
# Train the classifier
classifier = fasttext.supervised('./names/fasttext_data', 'fasttext_classifier', epoch=100000, lr=0.005)
fasttext_time = timeSince(start)

# Save accuracy for experiment comparison table
fasttext_prediction = classifier.predict(test_set_data)
fasttext_accuracy = np.mean(np.array(fasttext_prediction ) == np.array(test_set_labels))

rnn_prediction = [predict(i) for i in test_set_labels]
rnn_accuracy = np.mean(np.array(rnn_prediction) == np.array(test_set_labels))

# TODO: Conv_accuracy

with open ('classifiers_results', 'w') as f:
    f.write('Fast Text classification accuracy is: ' + str(fasttext_accuracy) + '\n')
    f.write('RNN classification accuracy is: ' + str(rnn_accuracy) + '\n')
    f.write('RNN training time: ' + str(rnn_time) + '\n')
    f.write('Fast Text training time ' + str(fasttext_time) + '\n')




