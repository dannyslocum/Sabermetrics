import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np


def decode_image(filename, image_type, resize_shape, channels=0):
    value = tf.read_file(filename)
    if image_type == 'png':
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif image_type == 'jpeg':
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)

    if resize_shape is not None and image_type in ['png', 'jpeg']:
            decoded_image = tf.image.resize_images(decoded_image, resize_shape)
        return decoded_image

def get_dataset(image_paths, image_type, resize_shape, channels):
    filename_tensor = tf.constant(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)
    def _map_fn(filename):
        return decode_image(filename, image_type, resize_shape, channels=channels)
    return dataset.map(_map_fn)

def get_image_data(image_paths, image_type=None, resize_shape=None, channels=0):
    dataset = get_dataset(image_paths, image_type, resize_shape, channels)
    iterator = dataset.make_one_shot_iterator()
    next_image = iterator.get_next()
    image_data_list = []
    with tf.Session() as sess:
        for i in range(len(image_paths)):
            image_data = sess.run(next_image)
            image_data_list.append(image_data)
    return image_data_list

def pil_resize_image(image_path, resize_shape,
    image_mode='RGBA', image_filter=None):
    im = Image.open(image_path)
    converted_im = im.convert(image_mode)
    resized_im = converted_im.resize(resize_shape, Image.LANCZOS)
    if image_filter is not None:
        resized_im = resized_im.filter(image_filter)
    return np.asarray(resized_im.getdata())



value = tf.read_file('image3.jpg')
with tf.Session() as sess:
    arr = sess.run(tf.image.decode_jpeg(value, channels=1))
    print(arr.shape)
    print(repr(arr))



image_paths = ['img1.jpg', 'img2.jpg']
dataset = get_image_data(image_paths)




class MNISTModel():
    def __init__(self, input_dim, output_size):
        self.input_dim = input_dim
        self.output_size = output_size

    def model_layers(self, inputs, is_training):
        reshaped_inputs = tf.reshape(inputs, [-1, self.input_dim, self.input_dim, 1])

        # convolutional Layer #1
        # pooling Layer #1
        conv1 = tf.layers.conv2d(
            inputs=reshaped_inputs,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
            name='conv1'
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2,
            name='pool1')

        # convolutional Layer #2
        # pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
            name='conv2')
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2,
            name='pool2')

        # fully-connected layer with activation RELU
        dense = self.create_fc(pool2)

        # apply dropout
        dropout = self.apply_dropout(dense, is_training)

        # get and return logits
        return self.get_logits(dropout)

    def create_fc(self, pool):
        hwc = pool.shape.as_list()[1:]
        flattened_size = hwc[0] * hwc[1] * hwc[2]
        pool_flat = tf.reshape(pool, [-1, flattened_size])
        return tf.layers.dense(pool_flat, 1024, activation=tf.nn.relu, name='dense')

    def apply_dropout(self, dense, is_training):
        return tf.layers.dropout(dense, rate=0.4, training=is_training)

    def get_logits(self, dropout):
        return tf.layers.dense(dropout, self.output_size, name='logits')

    def run_model_setup(self, inputs, labels, is_training):
        logits = self.model_layers(inputs, is_training)

        # convert logits to probabilities with softmax activation
        self.probs = tf.nn.softmax(logits, name='probs')

        # round probabilities
        self.predictions = tf.argmax(self.probs, axis=-1, name='predictions')
        class_labels = tf.argmax(labels, axis=-1)

        # find which predictions were correct
        is_correct = tf.equal(self.predictions, class_labels)
        is_correct_float = tf.cast(is_correct, tf.float32)

        # compute ratio of correct to incorrect predictions
        self.accuracy = tf.reduce_mean(is_correct_float)

        # train model
        if self.is_training:
            labels_float = tf.cast(labels, tf.float32)

            # compute the loss using cross_entropy
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_float, logits=logits)
            self.loss = tf.reduce_mean(cross_entropy)

            # use adam to train model
            adam = tf.train.AdamOptimizer()
            self.train_op = adam.minimize(self.loss, global_step=self.global_step)


class SqueezeNetModel(object):
    # Model Initialization
    def __init__(self, original_dim, resize_dim, output_size):
        self.original_dim = original_dim
        self.resize_dim = resize_dim
        self.output_size = output_size

    # Data Augmentation
    def image_preprocessing(self, data, is_training):
        reshaped_image = tf.reshape(data, [3, self.original_dim, self.original_dim])
        transposed_image = tf.transpose(reshaped_image, [1, 2, 0])
        float_image = tf.cast(transposed_image, tf.float32)
        if is_training:
            updated_image = self.random_crop_and_flip(float_image)
        else:
            updated_image = tf.image.resize_image_with_crop_or_pad(float_image, self.resize_dim, self.resize_dim)
        standardized_image = tf.image.per_image_standardization(updated_image)
        return standardized_image

    def random_crop_and_flip(self, float_image):
        crop_image = tf.random_crop(float_image, [self.resize_dim, self.resize_dim, 3])
        updated_image = tf.image.random_flip_left_right(crop_image)
        return updated_image

    # Convolution layer wrapper
    def custom_conv2d(self, inputs, filters, kernel_size, name):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            activation=tf.nn.relu,
            padding='same',
            name=name)

    def model_layers(self, inputs, is_training):
        conv1 = self.custom_conv2d(
            inputs,
            64,
            [3, 3],
            'conv1')
        pool1 = self.custom_max_pooling2d(
            conv1,
            'pool1')


    # SqueezeNet fire module
    def fire_module(self, inputs, squeeze_depth, expand_depth, name):
        with tf.variable_scope(name):
            squeezed_inputs = self.custom_conv2d(
                inputs,
                squeeze_depth,
                [1, 1],
                'squeeze')
            expand1x1 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [1, 1],
                'expand1x1')
            expand3x3 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [3, 3],
                'expand3x3')
            return tf.concat([expand1x1, expand3x3], axis=-1

    def multi_fire_module(self, layer, params_list):
        for params in params_list:
            layer = self.fire_module(
                layer,
                params[0],
                params[1],
                params[2]
            )
        return layer

    def model_layers(self, inputs, is_training):
        conv1 = self.custom_conv2d(inputs, 64, [3, 3], 'conv1')
        pool1 = self.custom_max_pooling2d(conv1, 'pool1')
        fire_params1 = [(32, 64, 'fire1'), (32, 64, 'fire2')]
        multi_fire1 = self.multi_fire_module(pool1, fire_params1)
        pool2 = self.custom_max_pooling2d(multi_fire1, 'pool2')
        fire_params2 = [(32, 128, 'fire3'), (32, 128, 'fire4')]
        multi_fire2 = self.multi_fire_module(pool2, fire_params2)
        dropout1 = tf.layers.dropout(multi_fire2, rate=0.5, training=is_training)
        final_conv_layer = self.custom_conv2d(dropout1, self.output_size, [1, 1], 'final_conv')
        return self.get_logits(final_conv_layer)

    def get_logits(self, conv_layer):
        avg_pool1 = tf.layers.average_pooling2d(conv_layer, [conv_layer.shape[1], conv_layer.shape[2]], 1)
        logits = tf.layers.flatten(avg_pool1, name='logits')
        return logits

    # Set up and run model training
    def run_model_setup(self, inputs, labels, is_training):
        logits = self.model_layers(inputs, is_training)
        self.probs = tf.nn.softmax(logits, name='probs')
        self.predictions = tf.argmax(self.probs, axis=-1, name='predictions')
        is_correct = tf.equal(tf.cast(self.predictions, tf.int32), labels)
        is_correct_float = tf.cast(is_correct, tf.float32)
        self.accuracy = tf.reduce_mean(is_correct_float)

        # calculate cross entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        self.loss = tf.reduce_mean(cross_entropy)
        adam = tf.train.AdamOptimizer()
        self.train_op = adam.minimize(self.loss, global_step=self.global_step)


block_layer_sizes = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}

"""
https://arxiv.org/pdf/1512.03385.pdf
"""
class ResNetModel(object):
    # Model Initialization
    def __init__(self, min_aspect_dim, resize_dim, num_layers, output_size,
        data_format='channels_last'):
        self.min_aspect_dim = min_aspect_dim
        self.resize_dim = resize_dim
        self.filters_initial = 64
        self.block_strides = [1, 2, 2, 2]
        self.data_format = data_format
        self.output_size = output_size
        self.block_layer_sizes = block_layer_sizes[num_layers]
        self.bottleneck = num_layers >= 50

    # Custom convolution function w/ consistent padding
    def custom_conv2d(self, inputs, filters, kernel_size, strides, name=None):
        if strides > 1:
            padding = 'valid'
            inputs = self.custom_padding(inputs, kernel_size)
        else:
            padding = 'same'
        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, data_format=self.data_format,
            name=name)

    def custom_padding(self, inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        if self.data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_before, pad_after], [pad_before, pad_after]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        return padded_inputs

    def pre_activation(self, inputs, is_training):
        axis = 1 if self.data_format == 'channels_first' else 3
        bn_inputs = tf.layers.batch_normalization(inputs, axis=axis, training=is_training)
        pre_activated_inputs = tf.nn.relu(bn_inputs)
        return pre_activated_inputs

    def pre_activation_with_shortcut(self, inputs, is_training, shortcut_params):
        pre_activated_inputs = self.pre_activation(inputs, is_training)
        shortcut = inputs
        shortcut_filters = shortcut_params[0]
        if shortcut_filters is not None:
            strides = shortcut_params[1]
            shortcut = self.custom_conv2d(pre_activated_inputs, shortcut_filters, 1, strides)
        return pre_activated_inputs, shortcut

    def regular_block(self, inputs, filters, strides, is_training, index, shortcut_filters=None):
        with tf.variable_scope('regular_block{}'.format(index)):
            shortcut_params = (shortcut_filters, strides)
            pre_activated1, shortcut = self.pre_activation_with_shortcut(inputs, is_training, shortcut_params)
            conv1 = self.custom_conv2d(pre_activated1, filters, 3, strides)
            pre_activated2 = self.pre_activation(conv1, is_training)
            conv2 = self.custom_conv2d(pre_activated2, filters, 3, 1)
            return conv2 + shortcut

    def bottleneck_block(self, inputs, filters, strides, is_training, index, shortcut_filters=None):
        with tf.variable_scope('bottleneck_block{}'.format(index)):
            shortcut_params = (shortcut_filters, strides)
            pre_activated1, shortcut = self.pre_activation_with_shortcut(inputs, is_training, shortcut_params)
            conv1 = self.custom_conv2d(pre_activated1, filters, 1, 1)
            pre_activated2 = self.pre_activation(conv1, is_training)
            conv2 = self.custom_conv2d(pre_activated2, filters, 3, strides)
            pre_activated3 = self.pre_activation(conv2, is_training)
            conv3 = self.custom_conv2d(pre_activated3, 4 * filters, 1, 1)
            return conv3 + shortcut

    def bottleneck_block(self, inputs, filters, strides, is_training, index, shortcut_filters=None):
        with tf.variable_scope('bottleneck_block{}'.format(index)):
            shortcut_params = (shortcut_filters, strides)
            pre_activated1, shortcut = self.pre_activation_with_shortcut(inputs, is_training, shortcut_params)
            conv1 = self.custom_conv2d(pre_activated1, filters, 1, 1)
            pre_activated2 = self.pre_activation(conv1, is_training)
            conv2 = self.custom_conv2d(pre_activated2, filters, 3, strides)
            pre_activated3 = self.pre_activation(conv2, is_training)
            conv3 = self.custom_conv2d(pre_activated3, 4 * filters, 1, 1)
            return conv3 + shortcut

    def model_layers(self, inputs, is_training):
        # initial convolution layer
        conv_initial = self.custom_conv2d(inputs, self.filters_initial, 7, 2, name='conv_initial')

        # pooling layer
        curr_layer = tf.layers.max_pooling2d(conv_initial, 3, 2, padding='same', data_format=self.data_format, name='pool_initial')

        # stack the block layers
        for i, num_blocks in enumerate(self.block_layer_sizes):
            filters = self.filters_initial * 2 ** i
            strides = self.block_strides[i]

            # stack this block layer on the previous one
            curr_layer = self.block_layer(curr_layer, filters, strides, num_blocks, is_training, i)

        # pre-activation
        pre_activated_final = self.pre_activation(curr_layer, is_training)
        filter_size = int(pre_activated_final.shape[2])

        # final pooling layer
        avg_pool = tf.layers.average_pooling2d(pre_activated_final, filter_size, 1, data_format=self.data_format)
        final_layer = tf.layers.flatten(avg_pool)

        # get logits from final layer
        return tf.layers.dense(final_layer, self.output_size, name='logits')
