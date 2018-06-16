import os
import sys
import click
import logging
import DCGAN
import tensorflow as tf

@click.group()
@click.version_option(0.1)
@click.option('-v', '--verbose', count=True)
@click.option('-l', '--log', default='last_run.txt')
def GANcontrol(verbose, log):
    """
    A utility script to generate population vectors out of neuro explorer raster data
    """
    if verbose >= 3:
        logconsolelevel = logging.DEBUG
    elif verbose >= 2:
        logconsolelevel = logging.INFO
    else:
        logconsolelevel = logging.INFO
 
    #Setup logging to file
    logging.basicConfig(level=logging.DEBUG, filename=log, filemode="w")

    #Add logging to the console
    console = logging.StreamHandler()
    console.setLevel(logconsolelevel)
    logging.getLogger('').addHandler(console)


@GANcontrol.command()
@click.argument('training_images')
@click.argument('epochs')
@click.argument('batch_size')
@click.argument('output_dir')
@click.option('--keep-checkpoints', 'keep_checkpoints', is_flag=True, help='Keep checkpoints')
def train(training_images, epochs, batch_size, output_dir, keep_checkpoints):
    """Train a DCGAN on given training images
    
    Arguments:
        training_images {str} -- [path to folder containing training images]
        epochs {int} -- [number of training cycles (batch_size images per cycle)]
        batch_size {int} -- [number of images to train per cycle]
        output_dir {str} -- [path to folder where to store results]
    """
    _train(training_images, epochs, batch_size, output_dir, keep_checkpoints)

def _train(training_images, epochs, batch_size, output_dir, keep_checkpoints):
    # Extract image size from image loading routine
    data = DCGAN.ISIC_data(training_images, randomize=False, seed=511)
    image_size = data.size

    # Seed tensorflow
    tf.set_random_seed(11223344)

    generator = DCGAN.G_conv(image_size=image_size)
    discriminator = DCGAN.D_conv(image_size=image_size)

	# run
    dcgan = DCGAN.DCGAN()
    dcgan.create(generator, discriminator, data, learning_rate=0.1e-4)
    dcgan.train(output_dir, training_epochs=epochs, checkpoints=epochs/10, batch_size=batch_size, keep_checkpoints=keep_checkpoints)


@GANcontrol.command()
@click.argument('model')
@click.argument('nimages')
@click.argument('output_dir')
@click.option('--overview', is_flag=True, help='Generate a single output file')
def generate(model, nimages, output_dir, overview):
    """Run a pretrained GAN to generate new images
    
    Arguments:
        model {str} -- [path to pretrained model]
        nimages {int} -- [number of images to generate]
        output_dir {str} -- [path to directory where to store images]
    """

    _generate(model, nimages, output_dir, overview)

def _generate(model, nimages, output_dir, overview):
    nimages = int(nimages)
    # Seed tensorflow
    tf.set_random_seed(11223344)

    dcgan = DCGAN.DCGAN()
    logging.info('Loading model from %s ...' % model)
    dcgan.load(model)

    logging.info('Generating %d images writing them to %s ...', nimages, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    dcgan.generate_images(nimages, output_dir, overview_figure=overview)


if __name__ == '__main__':
    GANcontrol()