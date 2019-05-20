from torch.utils.data import TensorDataset, DataLoader
import torch


def create_tensors(words, sequence_length):
    length = len(words) - sequence_length
    feature_tensors = []
    target_tensors = []
    for start in range(0, length):
        feature_tensors.append(words[start:sequence_length + start])
        target_tensors.append(words[sequence_length + start:sequence_length + start + 1])

    return torch.Tensor(feature_tensors), torch.Tensor(target_tensors)


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

print(create_tensors(words=[1, 2, 3, 4, 5, 6, 7], sequence_length=4))


data_loader = batch_data(words=[1, 2, 3, 4, 5, 6, 7], sequence_length=4, batch_size=2)





