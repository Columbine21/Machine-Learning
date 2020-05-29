import random
import time
import math


# Random item from a list
def randomChoice(array):
    return array[random.randint(0, len(array) - 1)]


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

