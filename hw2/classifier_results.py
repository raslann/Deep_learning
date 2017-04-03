from fastTextPreprocess import *
from model import *
from predict import predict
import numpy as np
import fasttext


classifier = fasttext.load_model('fasttext_classifier.bin', label_prefix=' __label__')

fasttext_prediction = classifier.predict(test_set_data)
fasttext_accuracy = np.mean(np.array(fasttext_prediction) == np.array(test_set_labels))

rnn_prediction = [predict(i) for i in test_set_labels]  #predict loads the rnn classifier and does the prediction
rnn_accuracy = np.mean(np.array(rnn_prediction) == np.array(test_set_labels))

with open ('classifiers_results', 'w') as f:
    f.write('Fast Text classification accuracy is: ' + str(fasttext_accuracy) + '\n')
    f.write('RNN classification accuracy is: ' + str(rnn_accuracy) + '\n')
    f.write('RNN training time: ' + str(rnn_time) + '\n')
    f.write('Fast Text training time ' + str(fasttext_time) + '\n')
