import tensorflow as tf
from graph_module import GraphModule
from tensorflow.python.framework import function



# # # # global gradient reversal functions  # # # #
def reverse_grad_grad(op, grad):
    return tf.constant(-1.) * grad


@function.Defun(tf.float32, python_grad_func=reverse_grad_grad)
def reverse_grad(tensor):
    return tf.identity(tensor)
# # # # # # # # # # # # # # # # # # # # # # # # # #



class FlippableClassifier(GraphModule):
    """ classifier with flippable gradients on the way out the bottom
    """
    def __init__(self, num_classes, reverse_grads, hidden_size=128, name='stacked_bidirectional'):
        super(FlippableClassifier, self).__init__(name)
        self.num_classes = num_classes
        self.reverse_grads = reverse_grads
        self.hidden_size = hidden_size


    def _build(self, labels, inputs, name='classifier'):
        if self.reverse_grads:
            input_shape = inputs.get_shape()
            inputs = reverse_grad(inputs)
            inputs.set_shape(input_shape)

        fc1 = tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=self.hidden_size,
            activation_fn=tf.nn.relu,
            scope='%s_fc' % name)
        logits = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=self.num_classes,
            activation_fn=None,
            scope='%s_pred' % name)

        # mean log perplexity
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        mean_loss = tf.reduce_mean(losses)

        return logits, mean_loss

