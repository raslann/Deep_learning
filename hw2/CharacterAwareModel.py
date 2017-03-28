
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as OPT
import numpy as np

from util import args, variable, var_to_numpy, glorot_uniform, orthogonal
from optim import RMSprop2

import numpy as NP
import numpy.random as RNG
import six

def noisify(x, std = .1):
    return x + variable(T.randn(x.size()) * std)

class ResLayer(NN.Module):
    
    def __init__(self, in_size, hidden_size):
        
        super(ResLayer, self).__init__()

        self.reshidden1 = NN.Linear(in_size, hidden_size)
        self.reshidden2 = NN.Linear(hidden_size, in_size)
        
        self.bn1 = NN.BatchNorm1d(hidden_size)
        self.bn2 = NN.BatchNorm1d(in_size)

    def forward(self, input_):
        '''
        @input_:

        '''
        
        fc1 = self.bn1(F.relu(noisify(self.reshidden1(input_), std=.4)))
        fc2 = self.bn2(F.relu(self.reshidden2(fc1)))
        
        return input_ + fc2



class CharacterAwareModel(NN.Module):
    def __init__(self, embed_size, state_size, vocab_size, output_size = None):
        if output_size == None:
            output_size = state_size
        super(CharacterAwareModel, self).__init__()

        self._embed_size = embed_size
        self._state_size = state_size
        self._vocab_size = vocab_size

        self.convHidden_size2 = embed_size//3
        self.convHidden_size3 = embed_size//3
        #Need to add up to the requested output size
        self.convHidden_size4 = embed_size - self.convHidden_size2 - self.convHidden_size3
        
        self.x = NN.Embedding(vocab_size, embed_size)
        
        self.conv2 = NN.Conv2d(1, self.convHidden_size2, kernel_size=2)
        self.conv3 = NN.Conv2d(1, self.convHidden_size3, kernel_size=3)
        self.conv4 = NN.Conv2d(1, self.convHidden_size4, kernel_size=4)
        
        convOutSize = self.convHidden_size2 + self.convHidden_size3 + self.convHidden_size4
        
        self._res_size = 100
        
        self.Res1 = ResLayer(convOutSize, self._res_size)
        self.Res2 = ResLayer(convOutSize, self._res_size)
        self.Res3 = ResLayer(convOutSize, self._res_size)
        
        

    def forward(self, input_):
        '''
        @input_:

        '''
        
        
        batch_size = len(input_)
        lengths = [len(wd)+2 for wd in input_]
        max_len = max(lengths)
        #ord('\a') = 7. This will be my buffer character.
        input_words = [chr(5) + wd + chr(6) + chr(7)*(max_len-len(wd)) for wd in input_]
        #input_words = [ord(chr) for wd in input_words for chr in wd]
        input_words =  [[ord(chr) for chr in wd] for wd in input_words]
    
        input_embedding = self.x(variable(T.LongTensor(input_words)))
        
        #[for [self.x(variable(T.LongTensor(char))) for char in input_words]

        #input 1 x 17 x 256. output 7 x 16 x 255
        c2 = F.relu(F.max_pool3d(self.conv2(input_embedding.unsqueeze(1)), [1,max_len+1,self._embed_size-1]))
        c3 = F.relu(F.max_pool3d(self.conv3(input_embedding.unsqueeze(1)), [1,max_len,self._embed_size-2]))
        c4 = F.relu(F.max_pool3d(self.conv4(input_embedding.unsqueeze(1)), [1,max_len-1,self._embed_size-3]))
        
        c2 = c2.resize(batch_size, self.convHidden_size2)
        c3 = c3.resize(batch_size, self.convHidden_size3)
        c4 = c4.resize(batch_size, self.convHidden_size4)
        
        conv_out = T.cat([c2, c3, c4],1)
        #conv_out = conv_out.resize(1,batch_size*3*(max_len+1)*(self._embed_size-1))
        
        
        
        fc1 = F.relu(self.Res1(conv_out))
        fc2 = F.relu(self.Res1(fc1))
        fc3 = F.relu(self.Res1(fc2))
        
        return fc3



class LanguageModel(NN.Module):
    def __init__(self, embed_size, state_size, vocab_size, vcb):
        super(LanguageModel, self).__init__()

        self._embed_size = embed_size
        self._state_size = state_size
        self._vocab_size = vocab_size
        
        self._vcb = vcb

        self.x = CharacterAwareModel(embed_size, state_size, 300)
        self.W = NN.LSTMCell(embed_size, state_size)
        self.W2 = NN.LSTMCell(state_size, state_size)
        self.W_y = NN.Linear(state_size, vocab_size + 1)    # 1 for <EOS>

    def forward(self, input_):
        '''
        @input_: LongTensor containing indices (batch_size, sentence_length)

        Return a 3D Tensor with size (batch_size, sentence_length, vocab_size)
        '''
        batch_size = input_.size()[0]
        length = input_.size()[1]

        h = variable(T.zeros(batch_size, self._state_size))
        c = variable(T.zeros(batch_size, self._state_size))
        h2 = variable(T.zeros(batch_size, self._state_size))
        c2 = variable(T.zeros(batch_size, self._state_size))

        output = []
        for t in range(0, length):
            hidden = [[input_[w,t]] for w in range(batch_size)]
            x = self.x.forward([self._vcb[input_[w,t].data[0]] for w in range(batch_size)])
            #x = self.x(input_[:, t])
            h, c = self.W(x, (h, c))
            h2, c2 = self.W2(h, (h2, c2))
            y = F.log_softmax(self.W_y(h2))
            output.append(y)

        return T.stack(output, 1)


def prepare_sentences(sentences):
    '''
    Transforms a batch of variable-length sentences into input
    tensor and mask tensor by padding and appropriate masking.

    Returns the maximum sentence length, input tensor, mask tensor
    and target tensor (integers).

    Note that the number of 1's in a mask row equals to the
    number of words in the corresponding sentence.
    '''
    # Pad them with zeros and build the mask matrix
    max_len = max([len(s) for s in sentences])

    input_ = T.zeros(args.batchsize, max_len).long()
    mask = T.zeros(args.batchsize, max_len).float()
    target = T.zeros(args.batchsize, max_len).long()

    for i in range(args.batchsize):
        input_[i, 0:len(sentences[i])] = T.LongTensor(sentences[i])
        mask[i, 0:len(sentences[i])] = 1

        # Set the target word to be the next word, and the target word
        # for the last word to be <EOS>
        if len(sentences[i]) > 1:
            target[i, 0:len(sentences[i])-1] = T.LongTensor(sentences[i][1:])
        target[i, len(sentences[i])-1] = vocab_size # <EOS>

    return max_len, input_, mask, target


def data_generator(tok, offsets, batch_size):
    dataset_size = len(offsets)
    num_batches = dataset_size // batch_size

    while True:
        sample_offsets = RNG.permutation(offsets)
        cur = 0

        for B in range(num_batches):
            sentences = []
            for i in range(batch_size):
                tokens = []
                while len(tokens) < args.minlength:
                    if cur == len(sample_offsets):
                        sample_offsets = RNG.permutation(offsets)
                        cur = 0
                    tok.seek(sample_offsets[cur])
                    cur += 1
                    tokens = tok.readline().strip().split()
                sentences.append([int(t) for t in tokens])
            yield prepare_sentences(sentences)


def clip_gradients(model, norm=1):
    grad_norm = 0
    for p in model.parameters():
        grad_norm = (p.grad.data ** 2).sum()
    grad_norm = NP.sqrt(grad_norm)
    if grad_norm > norm:
        for p in model.parameters():
            p.grad.data = p.grad.data / grad_norm * norm


if __name__ == '__main__':
    train_tok_name = args.trainname + '.tok'
    train_idx_name = args.trainname + '.idx'
    vocab_name = args.trainname + '.vcb'
    valid_tok_name = args.validname + '.tok'
    valid_idx_name = args.validname + '.idx'

    train_tok = open(train_tok_name, "r")
    train_idx = open(train_idx_name, "r")
    valid_tok = open(valid_tok_name, "r")
    valid_idx = open(valid_idx_name, "r")

    vocab_file = open(vocab_name, "r")
    vocab = [w.strip() for w in vocab_file.readlines()]
    vocab_size = len(vocab)
    vocab_file.close()

    model = LanguageModel(args.embedsize, args.statesize, vocab_size, vocab)
    if args.cuda:
        model.cuda()

    opt = OPT.Adam(model.parameters(), weight_decay=0.001)

    train_offsets = [int(l.strip()) for l in train_idx.readlines()]
    valid_offsets = [int(l.strip()) for l in valid_idx.readlines()]
    train_datagen = data_generator(train_tok, train_offsets, args.batchsize)
    valid_datagen = data_generator(valid_tok, valid_offsets, args.batchsize)
    train_batches = len(train_offsets) // args.batchsize
    valid_batches = len(valid_offsets) // args.batchsize

    for E in range(args.epochs):
        model.train()

        for B in range(train_batches):
            max_len, input_, mask, target = six.next(train_datagen)
            assert mask.sum(1).min() >= args.minlength

            input_ = variable(input_)
            mask = variable(mask)
            target = variable(target)
            output = model.forward(input_)
            masked_output = mask.unsqueeze(2).expand_as(output) * output
            masked_loss = -masked_output.gather(
                    2,
                    target.view(args.batchsize, max_len, 1)
                    )[:, :, 0]
            loss = masked_loss.sum() / mask.sum()

            loss.backward()

            clip_gradients(model, args.gradnorm)

            opt.step()

            six.print_('#%05d:%05d %-8.5f' % (E, B, var_to_numpy(loss)))

        model.eval()
        ppl = 0
        for B in range(valid_batches):
            max_len, input_, mask, target = six.next(valid_datagen)
            assert mask.sum(1).min() >= args.minlength

            # TODO: looks repetitive...
            input_ = variable(input_, volatile=True)
            mask = variable(mask, volatile=True)
            target = variable(target, volatile=True)
            output = model.forward(input_)
            masked_output = mask.unsqueeze(2).expand_as(output) * output
            masked_ppl = -masked_output.gather(
                    2,
                    target.view(args.batchsize, max_len, 1)
                    )[:, :, 0]
            ppl += NP.exp(var_to_numpy(masked_ppl.sum() / mask.sum()))
        ppl /= valid_batches

        six.print_('@%05d %-8.5f' % (E, ppl))
