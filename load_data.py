"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# load in data
import helper

data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)

view_line_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    words = tuple(set(text))
    int_to_vocab = dict(enumerate(words))
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    special_chars = dict()
    special_chars['.'] = "||Period||"
    special_chars[','] = "||Comma||"
    special_chars['"'] = "||Quotation_Mark||"
    special_chars[';'] = "||Semicolon||"
    special_chars['!'] = "||Exclamation_mark||"
    special_chars['?'] = "||Question_mark||"
    special_chars['('] = "||Left_Parentheses||"
    special_chars[')'] = "||Right_Parentheses||"
    special_chars['-'] = "||Dash||"
    special_chars['\n'] = "||Return||"

    return special_chars


helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')

from torch.utils.data import TensorDataset, DataLoader
import torch


def create_tensors(words, sequence_length):
    length = len(words) - sequence_length
    feature_tensors = []
    target_tensors = []
    for start in range(0, length):
        feature_tensors.append(words[start:sequence_length + start])
        target_tensors.append(words[sequence_length + start])

    return torch.tensor(feature_tensors, dtype=torch.long), torch.tensor(target_tensors, dtype=torch.long)


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function

    feature_tensors, target_tensors = create_tensors(words, sequence_length)

    data = TensorDataset(feature_tensors, target_tensors)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size)

    return data_loader


# test_text = range(50)
# t_loader = batch_data(test_text, sequence_length=5, batch_size=10)
#
# data_iter = iter(t_loader)
# sample_x, sample_y = data_iter.next()
#
# print(sample_x.shape)
# print(sample_x)
# print()
# print(sample_y.shape)
# print(sample_y)

sequence_length = 5  # of words in a sequence
# Batch Size
batch_size = 4

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)
inputs, labels = next(iter(train_loader))
# import torch.nn as nn
#
# n_vocab = len(int_to_vocab)
# n_embed = 500
# n_layers = 3
# embedding_dim = 500
# hidden_dim = 25
# dropout = 0.1
#
# embed = nn.Embedding(n_vocab, n_embed)
# lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
#                dropout=dropout, batch_first=True)
#
#
# out = embed(inputs)
# out = lstm.forward(out)
# print(out)

# print(lstm)
