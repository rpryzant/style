"""
"""
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.contrib.rnn.python.ops import rnn
from graph_module import GraphModule
import encoders
import decoders
import flippables
import numpy as np
from tf_utils import variable_summaries
import sys; sys.path.append('../')
from data.data_utils import prep_batch, prep_targets
import os

class Model:
    def __init__(self, sess, config, summary_dir):
        self.sess = sess
        self.config = config

        # dataset configs
        self.batch_size = config.batch_size
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size
        self.num_domains = config.num_domains

        # model configs
        self.discriminate = config.discriminate
        self.flip_grads = config.flip_grads
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.gradient_clip = config.gradient_clip
        self.train_dropout = config.train_dropout

        # TODO more stuff

        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='lr')
        self.dropout = tf.placeholder(tf.float32,  name='dropout')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.source = tf.placeholder(tf.int32, [self.batch_size, None], name='source')
        self.source_len = tf.placeholder(tf.int32, [self.batch_size], name='source_len')
        self.domains = tf.placeholder(tf.int32, [self.batch_size], name='domain_label')

        self.targets = tf.placeholder(tf.int32, [self.batch_size, None], name='target')
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name='target_len')

        # shared embeddings for everything (encoder, both decoders)
        self.embeddings = tf.get_variable('embeddings',
                    shape=[self.vocab_size, self.embedding_size])
        variable_summaries(self.embeddings, 'embeddings')

        source_encoding = self.encode(source=self.source, source_len=self.source_len)

        self.domain_logits, domain_loss = self.domain_classify(
            self.domains,
            source_encoding.h)  # give h vector to classifier (same as c)

        self.decode_logits, decoding_loss = self.decode(source_encoding)

        self.loss = decoding_loss + (domain_loss if self.discriminate else 0)

        tf.summary.scalar('decoding_loss', decoding_loss)
        tf.summary.scalar('domain_loss', domain_loss)
        tf.summary.scalar('global_loss', self.loss)

        self.train_step = self.optimize(self.loss)

        self.saver = tf.train.Saver()
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)



    def optimize(self, loss):
        """ creates a training op 
        """
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            clip_gradients=self.gradient_clip,
            optimizer='Adam',
            summaries=["learning_rate", "loss", "gradient_norm"])

        return train_op


    def decode(self, encoding):
        with tf.variable_scope('decoding'):

            with tf.variable_scope('d0_decoder'):
                d0_cell = self.build_rnn_cell(self.hidden_size * 2)  # encoder states were concatted
                d0_decoder = decoders.RawDecoder(
                    cell=d0_cell,
                    decoding_steps=self.target_len,
                    hidden_size=self.hidden_size,
                    output_size=self.vocab_size,
                    embeddings=self.embeddings,
                    batch_size=self.batch_size)

            with tf.variable_scope('d1_decoder'):
                d1_cell = self.build_rnn_cell(self.hidden_size * 2)  # encoder states were concatted
                d1_decoder = decoders.RawDecoder(
                    cell=d1_cell,
                    decoding_steps=self.target_len,
                    hidden_size=self.hidden_size,
                    output_size=self.vocab_size,
                    embeddings=self.embeddings,
                    batch_size=self.batch_size)

            # decode with the right decoder
            # TODO -- verify that this is working???
            is_d1 = tf.reduce_all( (self.domains > 0) )
            logits = tf.cond(is_d1, lambda: d1_decoder(encoding), lambda: d0_decoder(encoding))

            loss = self.cross_entropy_loss(logits, self.targets)

            return logits, loss


    def cross_entropy_loss(self, logits, targets):
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(targets, depth=self.vocab_size, dtype=tf.float32),
            logits=logits)
        return tf.reduce_mean(losses)


    def domain_classify(self, labels, inputs):
        """ classify source domain based on its encoding
        """
        with tf.variable_scope('clasification'):
            classifier = flippables.FlippableClassifier(
                num_classes=self.num_domains,
                reverse_grads=self.flip_grads,
                hidden_size=self.hidden_size)

            logits, loss = classifier(labels, inputs)

            return logits, loss


    def encode(self, source, source_len):
        """ encode a source sequence
        """
        source_embedded = self.get_embeddings(self.source)

        with tf.variable_scope('encoder'):
            with tf.variable_scope('cell_fw'):
                cell_fw = self.build_rnn_cell(self.hidden_size)
            with tf.variable_scope('cell_bw'):
                cell_bw = self.build_rnn_cell(self.hidden_size)

            if self.num_layers == 1:
                encoder = encoders.BidirectionalEncoder(cell_fw, cell_bw)
            else:
                encoder = encoders.StackedBidirectionalEncoder(cell_fw, cell_bw)

        outputs, final_state = encoder(source_embedded, source_len)

        return final_state


    def get_embeddings(self, source):
        """ looks up word embeddings for a source sequence
        """
        embedding = tf.nn.embedding_lookup(self.embeddings, source)
        return embedding


    def build_rnn_cell(self, hidden_size):
        """ build a stacked rnn cell
        """
        def build_cell():
            cell = tf.contrib.rnn.LSTMCell(hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                input_keep_prob=(1 - self.dropout))
            return cell

        if self.num_layers == 1:
            return build_cell()

        return tf.contrib.rnn.MultiRNNCell(
            [build_cell() for _ in range(self.num_layers)],
            state_is_tuple=True)


    def train_on_batch(self, seqs, domains, learning_rate=0.0005):
        """ NOTE: batch MUST be all from the SAME DOMAIN
        """
        x_batch, x_lens = prep_batch(seqs)
        y_batch, y_lens = prep_batch(prep_targets(seqs, self.config.wiggle))

        _, summaries, domain_logits, decode_logits, loss, step = self.sess.run(
            [self.train_step, self.summary_op, self.domain_logits, 
                self.decode_logits, self.loss, self.global_step],
            feed_dict={
                self.source: x_batch,
                self.source_len: x_lens,
                self.targets: y_batch,
                self.target_len: y_lens,
                self.domains: domains,
                self.dropout: self.train_dropout,
                self.learning_rate: learning_rate
            })

        self.summary_writer.add_summary(summaries, step)

        decode_preds = np.argmax(decode_logits, axis=2)
        domain_preds = np.argmax(domain_logits, axis=1)

        return loss, decode_preds, domain_preds


    def val_on_batch(self, seqs, domains):
        """ NOTE: batch MUST be all from the SAME DOMAIN
        """
        x_batch, x_lens = prep_batch(seqs)
        y_batch, y_lens = prep_batch(prep_targets(seqs, self.config.wiggle))

        domain_logits, decode_logits, loss, step = self.sess.run(
            [self.domain_logits, self.decode_logits, self.loss, self.global_step],
            feed_dict={
                self.source: x_batch,
                self.source_len: x_lens,
                self.targets: y_batch,
                self.target_len: y_lens,
                self.domains: domains,
                self.dropout: 0.0,
            })

        decode_preds = np.argmax(decode_logits, axis=2)
        domain_preds = np.argmax(domain_logits, axis=1)

        return loss, decode_preds, domain_preds
        


    def test_on_batch(self, seqs, domains):
        """ NOTE: batch MUST be all from the SAME DOMAIN
        """
        x_batch, x_lens = prep_batch(seqs)

        domain_logits, decode_logits= self.sess.run(
            [self.domain_logits, self.decode_logits],
            feed_dict={
                self.source: x_batch,
                self.source_len: x_lens,
                self.target_len: [x+self.config.wiggle for x in x_lens],
                self.domains: domains,
                self.dropout: 0.0,
            })

        decode_preds = np.argmax(decode_logits, axis=2)
        domain_preds = np.argmax(domain_logits, axis=1)

        return decode_preds, domain_preds


    def save(self, path):
        """ saves model params at path specified by "path"
        """ 
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, 'model')
        self.saver.save(self.sess, save_path, global_step=self.global_step)


    def load(self, filepath=None, dir=None):
        print('INFO: reading checkpoint...')
        if dir is not None:
            ckpt = tf.train.get_checkpoint_state(dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print 'INFO: success! model restored from %s' % ckpt.model_checkpoint_path
            else:
                raise Exception("ERROR: No checkpoint found at ", dir)
        elif filepath is not None:
            self.saver.restore(self.sess, filepath)
            print 'INFO: success! model restored from ', filepath
        else:
            raise Exception('ERROR: must provide a checkpoint filepath or directory')





if __name__ ==  '__main__':
    pass
    # c = Config()
    # with tf.Session() as sess:
    #     m = Model(sess, c)







