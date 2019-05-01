from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))
AugmentedTransition = namedtuple('AugmentedTransition', ('state', 'action', 'logprob', 'mask', 'next_state',
                                       'reward', 'done'))


class Memory(object):
    def __init__(self, transition='transition'):
        self.memory = []
        if transition == 'transition':
            self.transition = Transition
        elif transition == 'augmented_transition':
            self.transition = AugmentedTransition
        else:
            assert False

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return self.transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)


