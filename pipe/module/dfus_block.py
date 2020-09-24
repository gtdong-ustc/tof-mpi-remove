import tensorflow as tf
from activation import *
from conv import conv
def dfus_block(x, flg, regular, i):
    pref = 'dfus_block_' + str(i) + '_'

    # define initializer for the network
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
    block_filters = [
        24,
        6, 3,
        6, 3,
        8,
    ]
    filter_sizes = [
        1,
        3, 3,
        3, 3,
        1,
    ]

    dilation_sizes = [
        1,
        2, 1,
        1, 2,
        1,
    ]

    # change space
    ae_inputs = tf.identity(x, name='ae_' + pref + '_inputs')

    # convolutional layers: encoder

    conv_input = conv(pref=pref, inits=inits, current_input=ae_inputs, output_channel=block_filters[0],
                      filter_size=filter_sizes[0],dilation_rate=dilation_sizes[0], trainable=flg,
                      activation=relu, conv_num=0)
    conv_dilation_21_1 = conv(pref=pref, inits=inits, current_input=conv_input, output_channel=block_filters[1],
                      filter_size=filter_sizes[1],dilation_rate=dilation_sizes[1], trainable=flg,
                      activation=relu, conv_num=1)
    conv_dilation_21_2 = conv(pref=pref, inits=inits, current_input=conv_dilation_21_1, output_channel=block_filters[2],
                      filter_size=filter_sizes[2],dilation_rate=dilation_sizes[2], trainable=flg,
                      activation=relu, conv_num=2)
    conv_dilation_12_1 = conv(pref=pref, inits=inits, current_input=conv_input, output_channel=block_filters[3],
                      filter_size=filter_sizes[3],dilation_rate=dilation_sizes[3], trainable=flg,
                      activation=relu, conv_num=3)
    conv_dilation_12_2 = conv(pref=pref, inits=inits, current_input=conv_dilation_12_1, output_channel=block_filters[4],
                      filter_size=filter_sizes[4],dilation_rate=dilation_sizes[4], trainable=flg,
                      activation=relu, conv_num=4)

    tensor_input = tf.concat([conv_dilation_21_1, conv_dilation_21_2, conv_dilation_12_1, conv_dilation_12_2], axis=-1)

    conv_output = conv(pref=pref, inits=inits, current_input=tensor_input, output_channel=block_filters[5],
                      filter_size=filter_sizes[5],dilation_rate=dilation_sizes[5], trainable=flg,
                      activation=relu, conv_num=5)

    tensor_output = tf.concat([ae_inputs, conv_output], axis=-1)
    ae_outputs = tf.identity(tensor_output, name='ae_' + pref + '_outputs')
    return ae_outputs

def dfus_block_add_output_conv(x, flg, regular, i):
    pref = 'dfus_block_add_output_conv_' + str(i) + '_'

    # define initializer for the network
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
    block_filters = [
        24,
        6, 3,
        6, 3,
        8,
        1,
    ]
    filter_sizes = [
        1,
        3, 3,
        3, 3,
        1,
        1,
    ]

    dilation_sizes = [
        1,
        2, 1,
        1, 2,
        1,
        1,
    ]

    # change space
    ae_inputs = tf.identity(x, name='ae_' + pref + '_inputs')

    # convolutional layers: encoder

    conv_input = conv(pref=pref, inits=inits, current_input=ae_inputs, output_channel=block_filters[0],
                      filter_size=filter_sizes[0],dilation_rate=dilation_sizes[0], trainable=flg,
                      activation=relu, conv_num=0)
    conv_dilation_21_1 = conv(pref=pref, inits=inits, current_input=conv_input, output_channel=block_filters[1],
                      filter_size=filter_sizes[1],dilation_rate=dilation_sizes[1], trainable=flg,
                      activation=relu, conv_num=1)
    conv_dilation_21_2 = conv(pref=pref, inits=inits, current_input=conv_dilation_21_1, output_channel=block_filters[2],
                      filter_size=filter_sizes[2],dilation_rate=dilation_sizes[2], trainable=flg,
                      activation=relu, conv_num=2)
    conv_dilation_12_1 = conv(pref=pref, inits=inits, current_input=conv_input, output_channel=block_filters[3],
                      filter_size=filter_sizes[3],dilation_rate=dilation_sizes[3], trainable=flg,
                      activation=relu, conv_num=3)
    conv_dilation_12_2 = conv(pref=pref, inits=inits, current_input=conv_dilation_12_1, output_channel=block_filters[4],
                      filter_size=filter_sizes[4],dilation_rate=dilation_sizes[4], trainable=flg,
                      activation=relu, conv_num=4)

    tensor_input = tf.concat([conv_dilation_21_1, conv_dilation_21_2, conv_dilation_12_1, conv_dilation_12_2], axis=-1)

    conv_output = conv(pref=pref, inits=inits, current_input=tensor_input, output_channel=block_filters[5],
                      filter_size=filter_sizes[5],dilation_rate=dilation_sizes[5], trainable=flg,
                      activation=relu, conv_num=5)

    tensor_output = tf.concat([ae_inputs, conv_output], axis=-1)
    conv_final_output = conv(pref=pref, inits=inits, current_input=tensor_input, output_channel=block_filters[6],
                      filter_size=filter_sizes[6],dilation_rate=dilation_sizes[6], trainable=flg,
                      activation=None, conv_num=6)
    ae_outputs = tf.identity(conv_final_output, name='ae_' + pref + '_outputs')
    return ae_outputs