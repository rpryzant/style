import utils
import configure
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main(config, train_size=0.8, min_length=6, max_length=30):
    for era in ["historic", "modern"]:
        seqs = utils.load_pickle(config.all_sequences[era])
        n = len(seqs)
        seqs = [s for s in seqs if min_length <= len(s) <= max_length]
        print len(seqs), n

        lengths = [len(s) for s in seqs]
        #sns.distplot(lengths)
        #plt.show()

        random.shuffle(seqs)
        train_size = int(train_size * len(seqs))
        utils.write_pickle(seqs[:train_size], config.train_sequences[era])
        utils.write_pickle(seqs[train_size:], config.dev_sequences[era])


if __name__ == '__main__':
    main(configure.Config())
