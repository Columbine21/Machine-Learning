import os
import glob
import string
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import List, Tuple, Dict

from Utils import randomChoice, timeSince
from Rnn import RNN

def build_character_set(dataset: Dict[str, List]) -> List:
    """
    Methods to build our character set (all character in dataset & <eos> as the end token).

    Parameters
    -------
      dataset : Dict {category name : [name set]}

    Returns
    -------
      character_set : List (our character set)

    """
    _character_set = set()
    for _category in dataset.keys():
        for name in dataset[_category]:
            _character_set.update(name)
    _character_set = list(sorted(_character_set))
    _character_set.append('<eos>')
    return _character_set


def construct_dataset() -> Tuple[List, Dict[str, List]]:
    """
    Methods to construct our own dataset.

    Returns
    -------
      all_categories : List specifically ['male', 'female']

      dataset : Dict {category name : [name set]}
    """
    _all_categories = []
    _dataset = {}
    for filename in glob.glob("../data/*male.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        _all_categories.append(category)
        category_nameList = []
        with open(filename) as file:
            for line in file.readlines():
                if not line.startswith('# ') and len(line.strip()) > 0:
                    category_nameList.append(line.strip())
        _dataset[category] = category_nameList
    return _all_categories, _dataset


def randomTrainingPair(_all_categories: List[str], dataset: Dict[str, List]) -> Tuple[str, str]:
    # Get a random category and random line from that category
    selected_category = randomChoice(_all_categories)
    line = randomChoice(dataset[selected_category])
    return selected_category, line


def categoryTensor(_category: str, _all_categories: List[str]) -> torch.Tensor:
    # One-hot vector for category
    li = _all_categories.index(_category)
    tensor = torch.zeros(1, len(_all_categories))
    tensor[0][li] = 1
    return tensor


def inputTensor(name: str, _character_set: List) -> torch.Tensor:
    # One-hot matrix of first to last letters (not including EOS) for input
    tensor = torch.zeros(len(name), 1, len(_character_set))
    for index in range(len(name)):
        letter = name[index]
        tensor[index][0][_character_set.index(letter)] = 1
    return tensor


def targetTensor(name: str, _character_set: List) -> torch.Tensor:
    # LongTensor of second letter to end (EOS) for target
    letter_indexes = [_character_set.index(name[index]) for index in range(1, len(name))]
    letter_indexes.append(len(_character_set) - 1) # EOS
    return torch.LongTensor(letter_indexes)


def randomTrainingExample(_all_categories: List[str], dataset: Dict[str, List], _character_set: List) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Make category, input, and target tensors from a random category, line pair
    category, name = randomTrainingPair(_all_categories, dataset)
    category_tensor = categoryTensor(category, _all_categories)
    input_line_tensor = inputTensor(name, _character_set)
    target_line_tensor = targetTensor(name, _character_set)
    return category_tensor, input_line_tensor, target_line_tensor


if __name__ == "__main__":
    all_categories, nameSet = construct_dataset()
    # Todo: Test construct_dataset
    # print(all_categories)

    # for category in nameSet.keys():
    #     print(category, len(nameSet[category]))

    character_set = build_character_set(nameSet)
    # Todo: Test build_character_set
    # print(len(character_set), character_set)
    # print(randomTrainingPair(all_categories, nameSet))

    # Todo: Test function categoryTensor & inputTensor & targetTensor.
    # print(categoryTensor("female", all_categories))
    # print(categoryTensor("male", all_categories))
    # print(inputTensor("Jason", character_set))
    # print(targetTensor("Jason", character_set))

    # Todo: Test randomTrainingExample
    # print(randomTrainingExample(all_categories, nameSet, character_set))
    criterion = nn.NLLLoss()

    learning_rate = 0.0005

    rnn = RNN(len(all_categories), len(character_set), 128, len(character_set))

    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

    def train(category_tensor, input_line_tensor, target_line_tensor):
        target_line_tensor.unsqueeze_(-1)
        hidden = rnn.initHidden()

        rnn.zero_grad()

        loss = 0

        for i in range(input_line_tensor.size(0)):
            output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
            loss_tmp = criterion(output, target_line_tensor[i])
            loss += loss_tmp

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # for p in rnn.parameters():
        #     p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item() / input_line_tensor.size(0)


    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0  # Reset every plot_every iters

    start = time.time()

    for epochs in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample(all_categories, nameSet, character_set))
        total_loss += loss

        if epochs % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), epochs, epochs / n_iters * 100, loss))

        if epochs % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()
