
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

class CharacterModel(NN.Module):
    def __init__(self, embed_size, state_size, vocab_size, output_size = None):
        if output_size == None:
            output_size = state_size
        super(CharacterModel, self).__init__()

        self._embed_size = embed_size
        self._state_size = state_size
        self._vocab_size = vocab_size

        self.x = NN.Embedding(vocab_size, embed_size)
        self.fLSTM = NN.LSTMCell(embed_size, state_size)
        self.bLSTM = NN.LSTMCell(embed_size, state_size)
        self.fOut = NN.Linear(state_size, output_size)
        self.bOut = NN.Linear(state_size, output_size)

    def forward(self, input_):
        '''
        @input_: LongTensor containing indices (batch_size, sentence_length)

        '''
        batch_size = len(input_)
        lengths = [len(wd) for wd in input_]
        max_len = max(lengths)
        #ord('\a') = 7. This will be my buffer character.
        input_back = [wd[::-1] + chr(7)*(max_len-len(wd)) for wd in input_]
        input_forw = [wd + chr(7)*(max_len-len(wd)) for wd in input_]
        h = variable(T.zeros(batch_size, self._state_size))
        c = variable(T.zeros(batch_size, self._state_size))
        h2 = variable(T.zeros(batch_size, self._state_size))
        c2 = variable(T.zeros(batch_size, self._state_size))

        h_output = [None]*batch_size
        h2_output = [None]*batch_size
        for t in range(max_len):
            chars = [None]*batch_size
            for i in range(batch_size):
                chars[i] = ord(input_forw[i][t])
            f = self.x(variable(T.LongTensor(chars)))
            #I wanted to do the following line of code. but it didnt work so I used the above.
            #f = self.x(variable(T.LongTensor([ord(wd[t]) for wd in input_forw])))
            h, c = self.fLSTM(f, (h, c))
            chars = [None]*batch_size
            for i in range(batch_size):
                chars[i] = ord(input_back[i][t])
            b = self.x(variable(T.LongTensor(chars)))
            #Same as above
            #b = self.x(variable(T.LongTensor([ord(wd[t]) for wd in input_back])))
            h2, c2 = self.bLSTM(b, (h2, c2))
            for idx, l in enumerate(lengths):
                if l == t+1:
                    h_output[idx] = h[idx]
                    h2_output[idx] = h2[idx]
        h_output = T.stack(h_output, 0)
        h2_output = T.stack(h2_output, 0)
        return self.fOut(h_output) + self.bOut(h2_output)#T.stack(output, 1)



class LanguageModel(NN.Module):
    def __init__(self, embed_size, state_size, vocab_size, vcb):
        super(LanguageModel, self).__init__()

        self._embed_size = embed_size
        self._state_size = state_size
        self._vocab_size = vocab_size
        
        self._vcb = vcb

        self.x = CharacterModel(embed_size, state_size, 300)
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
