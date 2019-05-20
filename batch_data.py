from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function

    n_batches = len(words) // batch_size
    # only full batches
    words = words[:n_batches * batch_size]

    # TODO: Implement function
    features, targets = [], []

    for idx in range(0, (len(words) - sequence_length)):
        features.append(words[idx: idx + sequence_length])
        targets.append(words[idx + sequence_length])

    data = TensorDataset(torch.from_numpy(np.asarray(features)), torch.from_numpy(np.asarray(targets)))
    data_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size)

    return data_loader
    #
    # feature_tensors, target_tensors = create_tensors(words, sequence_length)
    #
    # data = TensorDataset(feature_tensors, target_tensors)
    # data_loader = torch.utils.data.DataLoader(data,
    #                                           batch_size=batch_size)
    #
    # return data_loader


# print(create_tensors(words=[1, 2, 3, 4, 5, 6, 7], sequence_length=4))


data_loader = batch_data(words=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], sequence_length=4, batch_size=3)
it = iter(data_loader)
x, y = next(it)
print(x)
print(y)
