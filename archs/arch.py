
import tensorflow as tf

from ops import conv1d_layer, conv1d_resnet_block, conv1d_resnet_block2

def build_graph(x_in, is_training, config):

    activation_fn = tf.nn.relu

    x_in_shp = tf.shape(x_in)

    cur_input = x_in
    print(cur_input.shape)
    idx_layer = 0
    numlayer = config.net_depth
    ksize = 1
    nchannel = config.net_nchannel
    # Use resnet or simle net
    act_pos = config.net_act_pos
    conv1d_block = conv1d_resnet_block
    conv1d_block2 = conv1d_resnet_block2

    # First convolution
    with tf.variable_scope("hidden-input"):
        cur_input = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=nchannel,
            activation_fn=None,
            perform_bn=False,
            perform_gcn=False,
            is_training=is_training,
            act_pos="pre",
            data_format="NHWC",
        )
        print(cur_input.shape)

    for _ksize, _nchannel in zip(
            [ksize] * (12), [nchannel] * (12)):

        with tf.variable_scope("output"):
            adjust = conv1d_layer(
                inputs=cur_input,
                ksize=1,
                nchannel=1,
                activation_fn=None,
                is_training=is_training,
                perform_bn=False,
                perform_gcn=False,
                data_format="NHWC",
            )
        adjust = tf.reshape(adjust, (x_in_shp[0], 1, x_in_shp[2], 1))

        scope_name = "hidden-" + str(idx_layer)
        with tf.variable_scope(scope_name):
            cur_input = conv1d_block2(
                inputs=cur_input,
                adjust=adjust,
                ksize=_ksize,
                nchannel=_nchannel,
                activation_fn=activation_fn,
                is_training=is_training,
                perform_bn=config.net_batchnorm,
                perform_gcn=config.net_gcnorm,
                act_pos=act_pos,
                data_format="NHWC",
            )
            # Apply pooling if needed
            print(cur_input.shape)

        idx_layer += 1

    with tf.variable_scope("output1"):
        logits_pre = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=1,
            activation_fn=None,
            is_training=is_training,
            perform_bn=False,
            perform_gcn=False,
            data_format="NHWC",
        )
        logits_pre = tf.reshape(logits_pre, (x_in_shp[0], x_in_shp[2]))
        print(logits_pre.shape)
        # adjust = tf.nn.sigmoid(logits_pre)
        adjust = tf.reshape(logits_pre, (x_in_shp[0], 1, x_in_shp[2], 1))

        for _ksize, _nchannel in zip(
                [ksize] * (3), [nchannel] * (3)):
            scope_name = "hidden-" + str(idx_layer)
            with tf.variable_scope(scope_name):
                cur_input = conv1d_block2(
                    inputs=cur_input,
                    adjust=adjust,
                    ksize=_ksize,
                    nchannel=_nchannel,
                    activation_fn=activation_fn,
                    is_training=is_training,
                    perform_bn=config.net_batchnorm,
                    perform_gcn=config.net_gcnorm,
                    act_pos=act_pos,
                    data_format="NHWC",
                )
                # Apply pooling if needed
                print(cur_input.shape)

            idx_layer += 1

        with tf.variable_scope("output2"):
            logits_pre2 = conv1d_layer(
                inputs=cur_input,
                ksize=1,
                nchannel=1,
                activation_fn=None,
                is_training=is_training,
                perform_bn=False,
                perform_gcn=False,
                data_format="NHWC",
            )
            logits_pre2 = tf.reshape(logits_pre2, (x_in_shp[0], x_in_shp[2]))
        print(logits_pre2.shape)
        # adjust = tf.nn.sigmoid(logits_pre)
        adjust2 = tf.reshape(logits_pre2, (x_in_shp[0], 1, x_in_shp[2], 1))

        for _ksize, _nchannel in zip(
                [ksize] * (3), [nchannel] * (3)):
            scope_name = "hidden-" + str(idx_layer)
            with tf.variable_scope(scope_name):
                cur_input = conv1d_block2(
                    inputs=cur_input,
                    adjust=adjust2,
                    ksize=_ksize,
                    nchannel=_nchannel,
                    activation_fn=activation_fn,
                    is_training=is_training,
                    perform_bn=config.net_batchnorm,
                    perform_gcn=config.net_gcnorm,
                    act_pos=act_pos,
                    data_format="NHWC",
                )
                # Apply pooling if needed
                print(cur_input.shape)

            idx_layer += 1

        with tf.variable_scope("output3"):
            cur_input = conv1d_layer(
                inputs=cur_input,
                ksize=1,
                nchannel=1,
                activation_fn=None,
                is_training=is_training,
                perform_bn=False,
                perform_gcn=False,
                data_format="NHWC",
            )
            logits = tf.reshape(cur_input, (x_in_shp[0], x_in_shp[2]))
        print(logits.shape)

        return logits, logits_pre, logits_pre2


