from model import *
from fastTextPreprocess import *
from train import *


rnn = torch.load('char-rnn-classification.pt')

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(line):
    output = evaluate(Variable(lineToTensor(line)))
    topv, topi = output.data.topk(1)
    category_index = topi[0][0]

    return  all_categories[category_index]
