
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as OPT

from util import args, variable, var_to_numpy, glorot_uniform, orthogonal
from optim import RMSprop2

import numpy as NP
import numpy.random as RNG
import six


class LanguageModel(NN.Module):
    def __init__(self, embed_size, state_size, vocab_size, dropout, tie_weights, GRUCell, multi_LSTM, nlayers):
        super(LanguageModel, self).__init__()

        self._embed_size = embed_size
        self._state_size = state_size
        self._vocab_size = vocab_size
        self._dropout = dropout
        self._tie_weight = tie_weights
        self._GRUCell = GRUCell
        self._nlayers = nlayers
        self._multi_LSTM = multi_LSTM


        self.drop = NN.Dropout(dropout)
        self.x = NN.Embedding(vocab_size, embed_size)
        self.W = NN.LSTM(embed_size, state_size, nlayers)
        self.W_y = NN.Linear(state_size, vocab_size + 1)    # 1 for <EOS>  #Decoder

        self.init_weights()

        if tie_weights:
            self.W_y.weight = self.x.weight

        if GRUCell:
            self.W = NN.GRUCell(embed_size, state_size)

    def init_weights(self):
        initrange = 0.1
        self.x.weight.data.uniform_(-initrange, initrange)
        self.W_y.bias.data.fill_(0)
        self.W_y.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (T.autograd.Variable(weight.new(self._nlayers, bsz, self._state_size).zero_()),
                    T.autograd.Variable(weight.new(self._nlayers, bsz, self._state_size).zero_()))

    def forward(self, input_, hidden):
        '''
        @input_: LongTensor containing indices (batch_size, sentence_length)

        Return a 3D Tensor with size (batch_size, sentence_length, vocab_size)
        '''
        if self._multi_LSTM:
            x = self.drop(self.x(input_))
            output, hidden = self.W(x.transpose(0, 1), hidden)
            output = self.drop(output)
            y = self.W_y((output.view(output.size(0) * output.size(1), output.size(2))))
            return y.view(output.size(0), output.size(1), y.size(1)), hidden

        else: #either: GRU Cell or LSTM Cell
            batch_size = input_.size()[0]
            length = input_.size()[1]

            h = variable(T.zeros(batch_size, self._state_size))
            c = variable(T.zeros(batch_size, self._state_size))

            output = []
            for t in range(0, length):
                x = self.x(input_[:, t])
                if self._GRUCell:
                    h, c = self.W(x, c)
                else: #LSTM Cell
                    h, c = self.W(x, (h, c))
                y = F.log_softmax(self.W_y(h))
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

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == T.autograd.Variable:
        return T.autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


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

    model = LanguageModel(args.embedsize, args.statesize, vocab_size, args.dropout, args.tie_weight, args.GRUCell, args.multi_LSTM, args.nlayers)
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
        hidden = model.init_hidden(args.batchsize)

        for B in range(train_batches):
            max_len, input_, mask, target = six.next(train_datagen)
            assert mask.sum(1).min() >= args.minlength

            input_ = variable(input_)
            mask = variable(mask)
            target = variable(target)
            hidden = repackage_hidden(hidden)
            output = model.forward(input_, hidden)
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
