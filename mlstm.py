import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

class MultiplicativeLSTMCell(RNNCell):
  """Multiplicative LSTM cell (mLSTM)

  Implementation is based on:
    https://arxiv.org/abs/1609.07959
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               activation=tf.tanh):

    """Initializes the parameters of mLSTM cell

    Args:
      num_units: int, the number of units in the mLSTM cell
      forget_bias: (optional) float, default 1.0, The initial bias of
        the forget gate which is used to reduce the scale of forgetting
        at the beginning of the training
      activation: (optional) Activation function of the inner states
    """

    super(MultiplicativeLSTMCell, self).__init__()

    self.num_units = num_units
    self.forget_bias = forget_bias
    self.activation = activation

  @property
  def state_size(self):
    return LSTMStateTuple(self.num_units, self.num_units)

  @property
  def output_size(self):
    return self.num_units

  def __call__(self, inputs, state, scope=None):
    prev_c, prev_h = state

    # equation (18)
    m_in = self._linear([inputs, prev_h], 2 * self.num_units)
    W_mx, W_mh = tf.split(m_in, 2, axis=1)
    m_t = W_mx * W_mh

    # split into lstm gates
    lstm_in = self._linear([inputs, m_t], 4 * self.num_units)
    i, j, f, o = tf.split(lstm_in, 4, 1)

    g = self.activation(j)

    new_c = tf.sigmoid(f + self.forget_bias) * prev_c + \
            tf.sigmoid(i) * g

    new_h = self.activation(new_c) * tf.sigmoid(o)

    return new_h, LSTMStateTuple(new_c, new_h)

  @staticmethod
  def _linear(inputs, out_dim, with_bias=True, scope=None):
    """Apply linear mapping to inputs by multiplying it with
      a projection (weight) matrix

    Args:
      inputs: (batch x dimension), [list of] tensor[s]
      out_dim: int, the number of output units
        (used as second dim for weight)
      scope: String, the name of the scope

    Returns:
      (batch, out_dim), Tensor representing the output of the
      linear mapping of the given inputs
    """

    if inputs is None:
      raise ValueError('inputs for _linear is not defined correctly.')

    if isinstance(inputs, (list, tuple)):
      inputs = [inputs]

    shapes = [x.get_shape().as_list() for x in inputs]
    total_input_sz = 0

    for shape in shapes:
      if len(shape) > 2:
        raise ValueError('inputs to _linear should be 2D. Given shape: %s', str(shape))
      elif shape[1] is None:
        raise ValueError('inputs dimension axis should be defined')
      else:
        total_input_sz += shape[1]

    with tf.variable_scope(scope or "Linear"):
      weight = tf.get_variable("W", shape=[total_input_sz, out_dim])
      if len(inputs) == 1:
        out = tf.matmul(inputs[0], weight)
      else:
        out = tf.matmul(tf.concat(inputs, 1), weight)
      if with_bias:
        bias = tf.get_variable('bias', shape=[out_dim])
        out = tf.nn.bias_add(out, bias)
      return out