import tensorflow as tf

def conv(previous_pref=None, x=None, output_channel=None, filter_size=None, dilation_rate=None, trainable=None, activation=None, conv_num=None):

    pref = previous_pref  + "conv_" + str(conv_num)

    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                # self define the initializer function
                from tensorflow.python.framework import dtypes
                from tensorflow.python.ops.init_ops import Initializer
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name= "input")

    # define the initializer
    name = pref
    if name + '_bias' in inits:
        bias_init = eval(name + '_bias()')
    else:
        bias_init = tf.zeros_initializer()
    if name + '_kernel' in inits:
        kernel_init = eval(name + '_kernel()')
    else:
        kernel_init = None
    # convolution

    current_output = tf.layers.conv2d(
        inputs=current_input,
        filters=output_channel,
        kernel_size=[filter_size, filter_size],
        dilation_rate=(dilation_rate, dilation_rate),
        padding="same",
        activation=activation,
        trainable=trainable,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name,
    )
    return current_output

def transpose_conv(pref=None, inits=None, current_input=None, output_channel=None, filter_size=None, strides=None,trainable=None, activation=None, conv_num=None):

    name = pref + "transpose_conv_" + str(conv_num)

    # define the initializer
    if name + '_bias' in inits:
        bias_init = eval(name + '_bias()')
    else:
        bias_init = tf.zeros_initializer()
    if name + '_kernel' in inits:
        kernel_init = eval(name + '_kernel()')
    else:
        kernel_init = None

    # convolution

    current_output = tf.layers.conv2d_transpose(
        inputs=current_input,
        filters=output_channel,
        kernel_size=[filter_size, filter_size],
        strides=(strides,strides),
        padding="same",
        activation=activation,
        trainable=trainable,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name,
    )
    return current_output
