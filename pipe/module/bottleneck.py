
def bottleneck(x, flg, regular, inits, i):
    pref = 'bottleneck_'
    block_filters = [
        64, 64, 256
    ]
    filter_sizes = [
        1, 3, 1
    ]
    skips = [
        False, False, True
    ]

    # change space
    ae_inputs = tf.identity(x, name='ae_' + pref + str(i) + '_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name='bottleneck_' + str(i) + '_input')
    # convolutional layers: encoder
    conv = []
    pool = [current_input]
    for i in range(1, len(block_filters)):
        name = pref + "conv_" + str(i)

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

        current_input = tf.layers.conv2d(
            inputs=current_input,
            filters=block_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            activation=relu,
            trainable=flg,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name,
        )
        current_input = tf.layers.batch_normalization(current_input, trainable=flg, training=flg, momentum=0.9)

        if skips[i] == True:
            shortcut = tf.layers.conv2d(
                inputs=ae_inputs,
                filters=block_filters[i],
                kernel_size=[filter_sizes[i], filter_sizes[i]],
                padding="same",
                activation=relu,
                trainable=flg,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
            current_input = current_input + shortcut
        current_input = relu(current_input)
        conv.append(current_input)

        return conv[-1]