import numpy as np
import random


PAD = 0
EOS = 1

def prep_targets(batch, wiggle=3):
    """ prepares autoencoding targets from inputs
        wiggle is how much wiggle room the decoder has
    """
    return [list(seq) + [EOS] + [PAD] * (wiggle - 1) for seq in batch]


def prep_batch(inputs, max_sequence_length=None):
    """
    args
        inputs: [ [ids] ]
        max seq len: if none, set to max len sequence

    returns:
        inputs: padded sequences
        lengths: length of each pre-padded sequence
    """
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_padded = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_padded[i, j] = element

    return inputs_padded, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]


def random_indices(batch_size, indices):
    while True:
        yield [
            random.choice(indices)
            for _ in range(batch_size)
        ]







