import random
import time
import math


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

