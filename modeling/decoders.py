import tensorflow as tf
from graph_module import GraphModule
from tensorflow.contrib.rnn.python.ops import rnn
from tf_utils import variable_summaries




class DynamicDecoder(GraphModule):

    def __init__(self, cell, decoding_steps, hidden_size, output_size, embeddings, batch_size, name='decoder'):
        super(DynamicDecoder, self).__init__(name)
        self.cell = cell
        self.batch_size = batch_size
        self.decoding_steps = decoding_steps
        self.hidden_size = hidden_size * 2   # encoder states were concatted
        self.output_size = output_size
        self.embedding_size = embedding_size

    def _build(self, encoding):
        # TODO -- use dynamic rnn
        return


class RawDecoder(GraphModule):
    """ multi-layer bidirectional encoders
    """
    def __init__(self, cell, decoding_steps, hidden_size, output_size, embeddings, batch_size, name='decoder'):
        super(RawDecoder, self).__init__(name)
        self.cell = cell
        self.batch_size = batch_size
        self.decoding_steps = decoding_steps
        self.hidden_size = hidden_size * 2   # encoder states were concatted
        self.output_size = output_size
        self.E = embeddings

        self.W = tf.get_variable(
            "projection_W",
            shape=[self.hidden_size, self.output_size],
            initializer=tf.contrib.layers.xavier_initializer())
        variable_summaries(self.W, 'projection_w')
        self.b = tf.Variable(tf.zeros([self.output_size], dtype=tf.float32))
        variable_summaries(self.b, 'projection_b')

        # NOTE - means index 0 must be reserved for 'pad'
        pad_batch = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')
        self.pad = tf.nn.embedding_lookup(self.E, pad_batch)

        # NOTE - means index 1 must reserved for '<EOS>'
        eos_batch = tf.ones([self.batch_size], dtype=tf.int32, name='EOS')
        self.eos = tf.nn.embedding_lookup(self.E, eos_batch)


    def _build(self, encoding):

        def transition(time, prev_out, prev_state, prev_loop_state):
            def next_input():
                logits = tf.add(tf.matmul(prev_out, self.W), self.b)
                pred = tf.argmax(logits, axis=1)
                next_inputs = tf.nn.embedding_lookup(self.E, pred)
                return next_inputs

            # test whether each batch is done
            is_finished = (time >= self.decoding_steps)
            #  see whether all baches are done
            finished = tf.reduce_all(is_finished)
            # next input is pad if done, else product of next_input()
            input = tf.cond(finished, lambda: self.pad, next_input)
            state = prev_state
            output = prev_out
            loop_state = None

            return (is_finished, input, state, output, loop_state)


        def loop(time, prev_out, prev_state, prev_loop_state):
            if prev_state is None:   # time is 0
                assert prev_out is None and prev_state is None
                return (
                    (0 >= self.decoding_steps),   # stopping conditions: all false
                    self.eos,                     # initial input: eos token
                    encoding,                     # given encoding 
                    None,                         # cell output
                    None)                         # loop state
            return transition(time, prev_out, prev_state, prev_loop_state)


        emitted_tensors, final_state, _ = tf.nn.raw_rnn(self.cell, loop)
        outputs = emitted_tensors.stack()

        T, B, H = tf.unstack(tf.shape(outputs))
        outputs_flat = tf.reshape(outputs, (-1, self.hidden_size)) # concat outputs across timesteps
        logits_flat = tf.add(tf.matmul(outputs_flat, self.W), self.b)
        logits = tf.reshape(logits_flat, (B, T, self.output_size))
        return logits



