""" Deep Convolutional Generative Adversarial Network (DCGAN).

Using deep convolutional generative adversarial networks (DCGAN) to generate
digit images from a noise distribution.

References:
    - Unsupervised representation learning with deep convolutional generative
    adversarial networks. A Radford, L Metz, S Chintala. arXiv:1511.06434.

Links:
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

%matplotlib inline

from PIL import Image
import itertools
import glob
import os
import numpy as np

def isic(image_path):
  # Load the images and convert them to numpy arrays
  images = []
  for img in glob.glob(os.path.join(image_path, '*.jpg')):        
    # Normalize between [0,1]
    images.append(np.array(Image.open(img))/255)
    
    # Normalize between [-1,1]
    #images.append(((np.array(Image.open(img))/255) - 0.5)*2.0)
  
  print('Loaded %d images ...' % (len(images)))
  return itertools.cycle(images)

def isic_next_batch(iterator, batch_size):
  return np.array([next(iterator) for i in range(batch_size)])

# Import ISIC data
ISIC = isic('./training_images')

# Training Params
num_steps = 100000
sample_output = 2000
batch_size = 64

# Network Params
sizeX = 50
sizeY = 50
sizeC = 3
image_dim = sizeX*sizeY*sizeC # 100*100 pixels * 3 channel
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 # Noise data points
G_learning_rate = 0.001
D_learning_rate = 0.001


# Generator Network
# Input: Noise, Output: Image
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        #print(x.get_shape())
        x = tf.layers.dense(x, units=6 * 6 * 196)
        #print(x.get_shape())
        x = tf.nn.tanh(x)
        #print(x.get_shape())
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 256)
        x = tf.reshape(x, shape=[-1, 6, 6, 196])
        #print(x.get_shape())
        # Deconvolution, image shape: (batch, 24, 24, 128)
        x = tf.layers.conv2d_transpose(x, 128, 14, strides=2)
        #print(x.get_shape())
        # Deconvolution, image shape: (batch, 28, 28, 1)
        #x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        
        # Deconvolution, image shape: (batch, 50, 50, 3)
        x = tf.layers.conv2d_transpose(x, sizeC, 4, strides=2)
        #print(x.get_shape())
        # Apply sigmoid to clip values between 0 and 1
        x = tf.nn.sigmoid(x)
        #print(x.get_shape())
        return x



# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5)
        #x = tf.layers.conv2d(x, 128, 5)
        #print(x.get_shape())
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
    return x

# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, sizeX, sizeY, sizeC])

# Build Generator Network
gen_sample = generator(noise_input)
#print('gen_sample size: %s' % (gen_sample.get_shape()))

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)

# Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Targets (real or fake images)
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

# Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=gen_target))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=G_learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=D_learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    f, a = plt.subplots(1, 1, figsize=(2*(sample_output/200), 10))
    all_images = None
    for i in range(1, num_steps+1):

        # Prepare Input Data
        # Get the next batch of MNIST data (only images are needed, not labels)
#        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = isic_next_batch(ISIC, batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, sizeX, sizeY, sizeC])
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Prepare Targets (Real image: 1, Fake image: 0)
        # The first half of data fed to the generator are real images,
        # the other half are fake images (coming from the generator).
        batch_disc_y = np.concatenate(
            [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
        # Generator tries to fool the discriminator, thus targets are 1.
        batch_gen_y = np.ones([batch_size])

        # Training
        feed_dict = {real_image_input: batch_x, noise_input: z, disc_target: batch_disc_y, gen_target: batch_gen_y}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict=feed_dict)
        if i % sample_output == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
            # Generate images from noise, using the generator network.
            
            for k in range(1):
                # Noise input.
                z = np.random.uniform(-1., 1., size=[5, noise_dim])
                g = sess.run(gen_sample, feed_dict={noise_input: z})
                
                g = np.concatenate(g, axis=1)    
            try:
              all_images = np.concatenate((all_images, g), axis=0)
            except ValueError:
              all_images = g.copy()
      
    a.imshow(all_images)
    a.axis('off')
    out_path = './out_images/full_run.jpg'
    f.savefig(out_path)
    print('Saved learning images to %s' % (out_path))