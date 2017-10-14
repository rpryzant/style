import utils
from operator import itemgetter


UNK = '<unk>'
START = '<start>'
END = '<end>'
NAME = '<name>'
PAD = '<pad>'
BLANK = '_'
SPECIAL_TOKENS = [PAD, START, END, UNK, NAME, BLANK]


class Vocabulary:
    def __init__(self, config, load=True, size=20000):
        self.config = config
        if load:
            self.tok_to_id = utils.load_pickle(config.tok_to_id)
        else:
            counts = utils.load_pickle(config.word_counts)
            words_and_counts = sorted(counts.items(), key=itemgetter(1), reverse=True)
            for (w, c) in words_and_counts[:20]:
                print w, c
            n = size - len(SPECIAL_TOKENS)
            head = sum(c for w, c in words_and_counts[:n])
            tail = sum(c for w, c in words_and_counts[n:])
            print words_and_counts[n]
            print head / float(head + tail), tail
            words = set(map(itemgetter(0), words_and_counts[:n]))

            self.tok_to_id = {}
            for token in SPECIAL_TOKENS:
                self.tok_to_id[token] = len(self.tok_to_id)
            for w in words:
                self.tok_to_id[w] = len(self.tok_to_id)
        self.id_to_tok = {i: w for w, i in self.tok_to_id.iteritems()}
        self.unknown = self.tok_to_id[UNK]
        self.start = self.tok_to_id[START]
        self.end = self.tok_to_id[END]
        self.name = self.tok_to_id[NAME]
        self.pad = self.tok_to_id[PAD]

    def add_word(self, w):
        self.tok_to_id[w] = len(self.tok_to_id)

    def __len__(self):
        return len(self.tok_to_id)

    def __contains__(self, w):
        if type(w) is str:
            return w in self.tok_to_id
        return w in self.id_to_tok

    def __getitem__(self, w):
        if type(w) is str:
            if w in self.tok_to_id:
                return self.tok_to_id[w]
            return UNK
        return self.id_to_tok[w]

    def write(self):
        utils.write_pickle(self.tok_to_id, self.config.tok_to_id)
