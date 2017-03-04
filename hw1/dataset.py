import torch as T
from torch.utils.data import TensorDataset, DataLoader
import pickle
import six


with open('train_labeled.p', 'rb') as f:
    train_labeled = pickle.load(f)
with open('train_unlabeled.p', 'rb') as f:
    train_unlabeled = pickle.load(f)
with open('validation.p', 'rb') as f:
    valid = pickle.load(f)
with open('test.p', 'rb') as f:
    test = pickle.load(f)

train_labeled_data = train_labeled.train_data
train_unlabeled_data = train_unlabeled.train_data
train_data = T.cat((train_labeled_data, train_unlabeled_data))
train_labeled_labels = train_labeled.train_labels
train_unlabeled_labels = T.zeros(train_unlabeled_data.size()[0]).long() - 1
train_labels = T.cat((train_labeled_labels, train_unlabeled_labels))

labeled_dataset = TensorDataset(train_labeled_data, train_labeled_labels)
unlabeled_dataset = TensorDataset(train_unlabeled_data, train_unlabeled_labels)
train_dataset = TensorDataset(train_data, train_labels)
valid_dataset = TensorDataset(valid.test_data, valid.test_labels)
test_dataset = TensorDataset(test.test_data, test.test_labels)
labeled_loader = DataLoader(labeled_dataset, 100, True)
unlabeled_loader = DataLoader(unlabeled_dataset, 100, True)
train_loader = DataLoader(train_dataset, 100, True)
valid_loader = DataLoader(valid_dataset, 100, True)
test_loader = DataLoader(test_dataset, 1, False)

labeled_gen = iter(labeled_loader)
unlabeled_gen = iter(unlabeled_loader)


def fetch():
    global labeled_gen, unlabeled_gen

    # Refresh
    if labeled_gen.samples_remaining == 0:
        labeled_gen = iter(labeled_loader)
    if unlabeled_gen.samples_remaining == 0:
        unlabeled_gen = iter(unlabeled_loader)

    return six.next(labeled_gen), six.next(unlabeled_gen)
