import utils
import configure
import vocabulary
import os
import re
import nltk
from operator import itemgetter
from collections import Counter


def normalize_word(w):
    return re.sub("\d", "0", w.lower())


def first_word(words, i):
    return i == 0 or words[i - 1] == '``'


def normalize_sentence(words, names, wordlike_names):
    for i, w in enumerate(words):
        if w.lower() in names and w[0].isupper() and \
                (w.lower() not in wordlike_names or not first_word(words, i)):
            words[i] = vocabulary.NAME
    return [normalize_word(w) for w in words]


def preprocess_sentence(sentence, vocab, names, wordlike_names):
    words = nltk.word_tokenize(sentence)
    words = normalize_sentence(words, names, wordlike_names)
    return [vocab[w] for w in words]


def sentences(config, normalize, include_historic=True, include_modern=True):
    if normalize:
        names = utils.load_pickle(config.names)
        wordlike_names = utils.load_pickle(config.wordlike_names)

    for decade in os.listdir(config.data_dir):
        decade_int = int(decade[:-1])
        if not ((include_historic and decade_int < 1860) or
                (include_modern and decade_int > 1980)):
            continue

        print "DECADE:", decade
        path = os.path.join(config.data_dir, decade)
        for fname in utils.logged_loop([fname for fname in os.listdir(path) if fname[:3] == "fic"]):
            with open(os.path.join(path, fname)) as f:
                for line in f:
                    line = line.strip().replace("<p>", "")
                    sentences = nltk.tokenize.sent_tokenize(line)
                    for s in sentences:
                        if "@" in s or "Page image" in s or "PRINTED" in s or "nbsp" in s:
                            continue
                        words = nltk.tokenize.word_tokenize(s)
                        if normalize:
                            words = normalize_sentence(words, names, wordlike_names)
                        yield words


def count_words(config, normalize):
    word_counts = Counter()
    capitalized_counts = Counter()
    for sentence in sentences(config, normalize):
        for i in range(len(sentence)):
            w = sentence[i]
            word_counts[w] += 1
            if w[0].isupper() and not first_word(sentence, i):
                capitalized_counts[w.lower()] += 1

    utils.write_pickle(word_counts, config.word_counts if normalize else config.word_counts_raw)
    if not normalize:
        utils.write_pickle(capitalized_counts, config.capitalized_counts)


def prep_data(config):
    vocab = vocabulary.Vocabulary(config)
    for era in ["historic", "modern"]:
        sequences = []
        for sentence in sentences(config, True,
                                  include_historic=(era == "historic"),
                                  include_modern=(era == "modern")):
            print " ".join([vocab[vocab[w]] for w in sentence])
            sequences.append([vocab[w] for w in sentence])
        utils.write_pickle(sequences, config.all_sequences[era])


def preprocess_names(config):
    word_counts = utils.load_pickle(config.word_counts_raw)
    capitalized_counts = utils.load_pickle(config.capitalized_counts)
    
    remove = ["english", "french", "german", "august", "president", "colonel", "lord", "june",
              "major", "states", "august", "sunday", "christmas", "america", "paris", "france",
              "florence", "roman", "israel", "ireland", "bible", "france", "england"]

    name_stats = []
    names, wordlike_names = set(), set()
    for name_file in [config.first_names_txt, config.last_names_txt]:
        with open(name_file) as f:
            for line in f:
                split = line.split()
                name = split[0].lower()
                freq = float(line.split()[1])

                count_ratio = word_counts[name] / max(1, float(capitalized_counts[name]))
                if name in remove and freq < 0.002:
                    print name
                if name not in remove and freq > 0.002:
                    name_stats.append((name, word_counts[name], capitalized_counts[name],
                                       count_ratio))
                    if count_ratio < 0.75:
                        names.add(name)
                        if count_ratio > 0.2:
                            wordlike_names.add(name)

    for w, wc, cc, cr in sorted(name_stats, key=itemgetter(-1), reverse=True):
        if cc > 500:
            print w, wc, cc, cr

    utils.write_pickle(names, config.names)
    utils.write_pickle(wordlike_names, config.wordlike_names)


def main(config):
    count_words(config, False)
    preprocess_names(config)
    count_words(config, True)
    vocabulary.Vocabulary(config, load=False).write()
    prep_data(config)

if __name__ == '__main__':
    main(configure.Config())
