"""
convenience class for reading in / batching up parsed data


"""
import cPickle
import random


class InputPipeline:

    def __init__(self, data_dir, batch_size):
        self.batch_size = batch_size

        data_dir += '/'

        print 'INFO: loading tok-id mapping...'
        self.tok_to_id = self.load_pickle(
            data_dir + 'word_vectors/tok_to_id.pkl')

        print 'INFO: loading modern sequences...'
        self.train_modern = self.replace_unks(self.load_pickle(
            data_dir + 'sequences/modern/train_sequences.pkl'))
        self.dev_modern = self.replace_unks(self.load_pickle(
            data_dir + 'sequences/modern/dev_sequences.pkl'))

        print 'INFO: loading historic sequences...'
        self.train_historic = self.replace_unks(self.load_pickle(
            data_dir + 'sequences/historic/train_sequences.pkl'))
        self.dev_historic = self.replace_unks(self.load_pickle(
            data_dir + 'sequences/historic/dev_sequences.pkl'))


        self.MODERN_DOMAIN = 0
        self.HISTORIC_DOMAIN = 1


    def load_pickle(self, fname):
        with open(fname) as f:
            return cPickle.load(f)


    def get_iterator(self, dataset='train'):
        if dataset == 'train':
            return self.data_iterator(self.train_modern, self.train_historic)
        return self.data_iterator(self.dev_modern, self.dev_historic)


    def replace_unks(self, seqs):
        return [[x  if type(x) == type(0) else self.tok_to_id[x] for x in s] for s in seqs]


    def data_iterator(self, modern, historic):
        modern_i = 0
        historic_i = 0
        while modern_i < len(modern) - self.batch_size and \
                historic_i < len(historic) - self.batch_size:
            if random.random() < 0.5:
                yield modern[modern_i: modern_i + self.batch_size],\
                      [self.MODERN_DOMAIN] * self.batch_size
                modern_i += self.batch_size
            else:
                yield historic[historic_i: historic_i + self.batch_size],\
                      [self.HISTORIC_DOMAIN] * self.batch_size
                historic_i += self.batch_size


    def vocab_size(self):
        return len(self.tok_to_id)

    def get_n(self, dataset='train'):
        if dataset == 'train':
            return len(self.train_historic) + len(self.train_modern)
        return len(self.dev_historic) + len(self.dev_modern)

    def num_batches(self, dataset='train'):
        return float(self.get_n(dataset)) / self.batch_size

