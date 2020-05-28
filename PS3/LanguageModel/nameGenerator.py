import os
import glob
import string

from typing import List, Tuple, Dict

from .Utils import randomChoice
from .Rnn import RNN



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








if __name__ == "__main__":
    all_categories, nameSet = construct_dataset()
    print(all_categories)

    for category in nameSet.keys():
        print(category, len(nameSet[category]))
    character_set = build_character_set(nameSet)
    print(character_set)


