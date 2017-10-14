import utils
from os.path import join


class Config(object):
    def __init__(self):
        # hyperparameters
        self.learning_rate = 0.5
        self.learning_rate_decay_factor = 0.99
        self.max_gradient_norm = 5
        self.batch_size = 64
        self.size = 1024
        self.num_layers = 2
        self.max_train_data_size = 0
        self.steps_per_checkpoint = 300
        self.eval_steps = 1
        self.eval_batch_size = 64

        # directories
        training_dir = './data'
        self.pretrained_vectors = None  # '/scr/nlp/data/wordvectors/en/glove/vectors_100d.txt'
        self.data_dir = '/scr/corpora/COHA/COHA text'

        names_dir = join(training_dir, 'names')
        self.first_names_txt = join(names_dir, 'first_names.txt')
        self.last_names_txt = join(names_dir, 'last_names.txt')
        self.names = join(names_dir, 'names.pkl')
        self.wordlike_names = join(names_dir, 'wordlike_names.pkl')

        preprocess_dir = join(training_dir, 'preprocessing')
        self.word_counts_raw = join(preprocess_dir, 'word_counts_raw.pkl')
        self.capitalized_counts = join(preprocess_dir, 'capitalized_counts.pkl')
        self.word_counts = join(preprocess_dir, 'word_counts.pkl')

        vectors_dir = join(training_dir, 'word_vectors')
        self.vectors = join(vectors_dir, 'word_vectors.npy')
        self.tok_to_id = join(vectors_dir, 'tok_to_id.pkl')

        sequences_dir = join(training_dir, 'sequences')
        historic_dir = join(sequences_dir, 'historic')
        modern_dir = join(sequences_dir, 'modern')
        self.all_sequences = {"historic": join(historic_dir, 'all_sequences.pkl'),
                              "modern": join(modern_dir, 'all_sequences.pkl')}
        self.train_sequences = {"historic": join(historic_dir, 'train_sequences.pkl'),
                                "modern": join(modern_dir, 'train_sequences.pkl')}
        self.dev_sequences = {"historic": join(historic_dir, 'dev_sequences.pkl'),
                              "modern": join(modern_dir, 'dev_sequences.pkl')}

        model_dir = join(training_dir, 'model')

        utils.mkdir(training_dir)
        utils.mkdir(preprocess_dir)
        utils.mkdir(vectors_dir)
        utils.mkdir(sequences_dir)
        utils.mkdir(historic_dir)
        utils.mkdir(modern_dir)
        utils.mkdir(model_dir)
