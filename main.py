import tensorflow as tf
from modeling.model import Model
from data.data_utils import prep_batch, random_sequences, prep_targets, random_indices
from data.input_pipeline import InputPipeline
import random
import argparse
import os
from msc.utils import Progbar

class Config:
    embedding_size = 128
    batch_size = 128
    max_len = 50
    vocab_size = 10
    num_domains = 2

    wiggle = 3

    flip_grads = False
    discriminate = True
    hidden_size = 128
    num_layers = 1
    gradient_clip = 5.

    train_dropout = 0.5
    epochs = 15


def process_command_line():
    """
    Return a 1-tuple: (args list).
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description='Usage') # add description
    # positional arguments
    parser.add_argument('data_dir', metavar='data_dir', type=str, 
        help='data directory')
    parser.add_argument('out_dir', metavar='out_dir', type=str, 
        help='output directory for checkpoints + summaries')

    # optional arguments
    parser.add_argument('-g', '--gpu', dest='gpu', type=str, default='0', 
        help='which gpu to run on')
    parser.add_argument('-ignore', '--ignore_discriminator', action='store_true',
        help='ignore the discriminator')
    parser.add_argument('-flip', '--flip_grads', action='store_true', 
        help='flip domain prediction gradients')


    args = parser.parse_args()
    return args



def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    c = Config()
    c.flip_grads = args.flip_grads
    c.discriminate = not args.ignore_discriminator

    print 'INFO: loading data...'
    input_pipeline = InputPipeline(args.data_dir, c.batch_size)
    c.vocab_size = input_pipeline.vocab_size()
    print 'INFO: done. train size: ', input_pipeline.get_n('train'), ' val: ', input_pipeline.get_n('val')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sess = tf.Session()
    m = Model(sess, c, args.out_dir + '/summaries/')
    sess.run(tf.global_variables_initializer())

    for epoch in range(c.epochs):
        print 'TRAIN: starting epoch ', epoch
        prog = Progbar(target=input_pipeline.num_batches('train'))
        train_iterator = input_pipeline.get_iterator('train')
        epoch_loss = 0
        i = 0
        for (seqs, domains) in train_iterator:
            loss, decode_preds, domain_preds = m.train_on_batch(seqs, domains)
            prog.update(i+1, [('train loss', loss)])
            epoch_loss += loss
            i += 1
        print 'TRAIN: done. avg loss: ', epoch_loss / i

        print 'TRAIN: saving checkpoint..'   
        m.save(args.out_dir + '/checkpoints/')

        print 'VAL: starting'
        prog = Progbar(target=input_pipeline.num_batches('val'))
        val_iterator = input_pipeline.get_iterator('val')
        epoch_loss = 0
        i = 0
        for (seqs, domains) in val_iterator:
            loss, decode_preds, domain_preds = m.val_on_batch(seqs, domains)
            prog.update(i+1, [('val loss', loss)])
            epoch_loss += loss
            i += 1
        print 'VAL: done. avg loss: ', epoch_loss / i










    quit()
    # BELOW THIS IS TESTING WITH RANDOM DATA

    # todo - constants file, make this pretty, config, command line args
    EOS = 1
    PAD = 0
    with tf.Session() as sess:
        c = Config()
        c.vocab_size = input_pipeline.vocab_size()
        m = Model(sess, c, 'summaries')


        sess.run(tf.global_variables_initializer())

        domains = [0, 1]

        batches = [next(random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=2))
                    for _ in range(10)]
        domains = [
            [random.choice(domains)] * 2
#            [1] * 2
            for _ in range(10)
        ]
        epoch_loss = 0
        print m.test_on_batch(batches[0], [0,0])
        quit()
        i = 0
        while True:
            d_in = domains[i]
            loss, decode_preds, domain_preds = m.train_on_batch(batches[i], d_in)
            epoch_loss += loss

            if i >= len(batches) - 1:
                i = 0
                print float(epoch_loss) / len(batches)
                epoch_loss = 0
            else:
                i += 1

        x_in, x_len = prep_batch(b)
        y_in, y_len = prep_batch(prep_targets(b, c.wiggle))



        i = 0
        while True:
            loss, decode_preds, domain_preds = m.train_on_batch(x_in, x_len, y_in, y_len, d_batch)
            print i, loss
            print x_in
            print decode_preds
            print d_batch
            print domain_preds
            print
            i += 1


if __name__ == '__main__':
    args = process_command_line()
    main(args)


