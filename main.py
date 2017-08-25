import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
import scipy.misc

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    
    vgg_input_tensor_name      = 'image_input:0'
    vgg_keep_prob_tensor_name  = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = sess.graph
    
    input_t      = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_t  = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_t = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_t = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_t = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_t, keep_prob_t, layer3_out_t, layer4_out_t, layer7_out_t
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    
    # Add the 1x1 layers
    out = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer)
    l4  = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer)
    l3  = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer)
    
    # Upsampling
    out = tf.layers.conv2d_transpose(out, filters=num_classes, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer)

    # Skip-layer + upsampling
    out = tf.add(out, l4)
    out = tf.layers.conv2d_transpose(out, filters=num_classes, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer)
    
    # Skip-layer + upsampling
    out = tf.add(out, l3)
    out = tf.layers.conv2d_transpose(out, filters=num_classes, kernel_size=16, strides=8, padding='same', kernel_initializer=initializer)
    
    return out
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Reshape to compute cross entropy
    logits        = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    
    # Loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    
    # Training op
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    for epoch_i in range(epochs):
        # The training cycle
        total_loss = []
        batch_i = 1
        for images, gt_images in get_batches_fn(batch_size):
            # Run optimizer and get loss
            _, l = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={input_image: images, correct_label: gt_images, learning_rate: 0.0001, keep_prob: 0.5})
                          
            print('Epoch {:2>}/{} - Batch {:2>} - Loss {}'.format(epoch_i+1, epochs, batch_i, l))
            batch_i +=1
            total_loss.append(l)
        
        mean_loss = sum(total_loss) / len(total_loss)
        print("\n Epoch {:2>}/{} - MEAN LOSS : {}\n".format(epoch_i+1, epochs, mean_loss))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)


    correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
    learning_rate = tf.placeholder(tf.float32)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        input_t, keep_prob_t, layer3_out_t, layer4_out_t, layer7_out_t = load_vgg(sess, vgg_path)
        model = layers(layer3_out_t, layer4_out_t, layer7_out_t, num_classes)
        logits, train_op, cross_entropy_loss = optimize(model, correct_label, learning_rate, num_classes)

        # Init global var
        sess.run(tf.global_variables_initializer())

        # Train NN using the train_nn function
        train_nn(sess, 20, 8, get_batches_fn, train_op, cross_entropy_loss, input_t, correct_label, keep_prob_t, learning_rate)
             
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_t, input_t)

if __name__ == '__main__':
    run()
