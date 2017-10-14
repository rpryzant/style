import tensorflow as tf
from graph_module import GraphModule
from tensorflow.contrib.rnn.python.ops import rnn
from tensorflow.contrib.rnn import LSTMStateTuple



class StackedBidirectionalEncoder(GraphModule):
    """ multi-layer bidirectional encoders
    """
    def __init__(self, cell_fw, cell_bw, name='stacked_bidirectional'):
        super(StackedBidirectionalEncoder, self).__init__(name)
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw

    def _build(self, inputs, lengths):
        outputs, final_fw_state, final_bw_state = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=self.cell_fw._cells,
            cells_bw=self.cell_bw._cells,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32)

        # Concatenate states of the forward and backward RNNs
        final_state = final_fw_state, final_bw_state

        return outputs, final_state



class BidirectionalEncoder(GraphModule):
    """ single-layer bidirectional encoder
    """
    def __init__(self, cell_fw, cell_bw, name='bidirectional'):
        super(BidirectionalEncoder, self).__init__(name)
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw

    def _build(self, inputs, lengths):
        outputs_pre, (final_fw_state, final_bw_state) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell_fw,
            cell_bw=self.cell_bw,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32)

        # concat c and h vectors from fw and bw lstms
        final_state_c = tf.concat(
            (final_fw_state.c, final_bw_state.c), 1)

        final_state_h = tf.concat(
            (final_fw_state.h, final_bw_state.h), 1)

        final_state = LSTMStateTuple(
            c=final_state_c,
            h=final_state_h
        )

        # Concatenate outputs of the forward and backward RNNs
        outputs = tf.concat(outputs_pre, 2)

        return outputs, final_state


