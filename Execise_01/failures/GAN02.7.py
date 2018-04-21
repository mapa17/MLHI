# Originally from: https://github.com/YadiraF/GAN/blob/master/dcgan.py

import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

import glob
import itertools
from PIL import Image


def leaky_relu(x, alpha=0.2):
	return tf.maximum(tf.minimum(0.0, alpha * x), x)

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
	return tf.random_normal(shape=size, stddev=xavier_stddev)

def write_figure(fig_path, epoch, data):
    for i, img in enumerate(data):
        fig, ax = plt.subplots(1, 1)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
        op = os.path.join(fig_path, "epoch_%06d_image_%03d.png" % (epoch, i))
        fig.savefig(op, bbox_inches='tight')
        plt.close(fig)

class G_conv(object):
	def __init__(self):
		self.name = 'G_conv'
		#self.size = 64/16
		#self.size = 4
		self.size = 20
		self.channel = 3

	def __call__(self, z):
		with tf.variable_scope(self.name) as scope:
			g = tcl.fully_connected(z, self.size * self.size * 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			g = tf.reshape(g, (-1, self.size, self.size, 1024))  # size
			g = tcl.conv2d_transpose(g, 512, 3, stride=2, # size*2
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.conv2d_transpose(g, 256, 3, stride=2, # size*4
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.conv2d_transpose(g, 128, 3, stride=2, # size*8
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			
			g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
										activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			return g
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv(object):
	def __init__(self):
		self.name = 'D_conv'

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			size = 320
			#size = 64
			shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
						stride=2, activation_fn=lrelu)
			shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=4, # 4x4x512
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

			shared = tcl.flatten(shared)
	
			d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
			return d, q
			
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

class DCGAN():
    def __init__(self, generator, discriminator, data, learning_rate=2e-4):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        # data
        self.z_dim = self.data.z_dim
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        # nets
        self.G_sample = self.generator(self.z)

        self.D_real, _ = self.discriminator(self.X)
        self.D_fake, _ = self.discriminator(self.G_sample, reuse = True)

        # loss
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.G_loss, var_list=self.generator.vars)

        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, sample_dir, DvsG_steps=5, GperD_steps=1, ckpt_dir='ckpt', training_epochs = 1000000, batch_size = 32, output_size=10000):
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            # update D
            for _ in range(DvsG_steps):
                X_b = self.data(batch_size)
                self.sess.run(self.D_solver,feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
                
            # update G
            for _ in range(GperD_steps):
                self.sess.run(self.G_solver, feed_dict={self.z: sample_z(batch_size, self.z_dim)})

            # save img, model. print loss
            if (epoch % output_size == 0) or (epoch == 1) or (epoch == training_epochs-1):
                D_loss_curr = self.sess.run(self.D_loss, feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
                G_loss_curr = self.sess.run(self.G_loss, feed_dict={self.z: sample_z(batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))
                samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(16, self.z_dim)})

                fig = self.data.data2fig(samples)
                plt.savefig('%s/epoch_%07d.png' % (sample_dir, epoch), bbox_inches='tight')
                plt.close(fig)
            
                write_figure(os.path.join(sample_dir, 'single_images'), epoch, samples)

				#if epoch % 2000 == 0:
					#self.saver.save(self.sess, os.path.join(ckpt_dir, "dcgan.ckpt"))


class ISIC_data():
    def __init__(self, training_data, size):
        self.z_dim = 100 # Size of the random input noise vector
        self.size = size
        self.channel = 3
        self.batch_count = 0

        self.data = glob.glob(os.path.join(training_data, '*.jpg'))
        self.images = []
        for img in self.data: 
            # Normalize between [0,1]
            #self.images.append((np.array(Image.open(img))/255).astype(np.float32))
            self.images.append((np.array(Image.open(img))/255.0))

            # Normalize between [-1,1]
            #images.append(((np.array(Image.open(img))/255.0) - 0.5)*2.0)

        print('Loaded %d images ...' % (len(self.images)))
        self.iterator = itertools.cycle(self.images)


    def __call__(self, batch_size):
        return np.array([next(self.iterator) for i in range(batch_size)])

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample)
        return fig


if __name__ == '__main__':
	# save generated images
    if len(sys.argv) > 1:
        sample_dir = sys.argv[1]
        epochs = int(sys.argv[2])
    else: 
        sample_dir = 'training_progress/'
        epochs = 1000
    print('Storing training progress in %s ...' % sample_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(os.path.join(sample_dir, 'single_images')):
        os.makedirs(os.path.join(sample_dir, 'single_images'))

	# param
    generator = G_conv()
    discriminator = D_conv()

    data = ISIC_data('./training_images_320x320_augmented_10k', 320)

	# run
    dcgan = DCGAN(generator, discriminator, data, learning_rate=0.00001)
    dcgan.train(sample_dir, training_epochs=epochs, output_size=epochs/10, batch_size=16)
