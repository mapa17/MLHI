import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster, metrics, utils
import sys
from pudb import set_trace as st
from sklearn import manifold
import itertools
import glob
import logging
import click
import os

from PIL import Image

@click.group()
@click.version_option(0.1)
@click.option('-v', '--verbose', count=True)
@click.option('-l', '--log', default='last_run.txt')
def img2dist(verbose, log):
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


def color_quantization(image_path, ncolors, sample_points=0.30):
    """Calculate color centroids using kmeans
    
    Arguments:
        image_path {[type]} -- [path to image]
        ncolors {[type]} -- [number of centroids/colors]
    
    Keyword Arguments:
        sample_points {float} -- [Ratio of pixels to use of kmeans] (default: {0.30})
    
    Returns:
        [numpy.array[floats]] -- [numpy array of size **ncolors**, 4]
    """
    # Load the Summer Palace photo
    img = Image.open(image_path)

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    img = np.array(img, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(img.shape)
    assert d == 3
    image_array = np.reshape(img, (w * h, d))
    nsample_points = int(w*h*sample_points)

    image_array_sample = sklearn.utils.shuffle(image_array, random_state=0)[:nsample_points]
    kmeans = sklearn.cluster.KMeans(n_clusters=ncolors, random_state=0).fit(image_array_sample)
    centroids = kmeans.cluster_centers_

    # Add centroid weight (color distribution)
    cnts, _ = np.histogram(kmeans.predict(image_array), bins=ncolors)
    cnts = cnts / cnts.sum()

    #points = np.concatenate([centroids, cnts[:, None]], axis=1)
    points = centroids
    return points 


def centroid_distance(centroids1, centroids2):
    """Calculate the distance between two sets of centroids.
    Sum of the pairwise distance between the closest centroids.
    
    Arguments:
        centroids1 {[numpy.array]} -- [array of centroids in 4 dimensions]
        centroids2 {[numpy.array]} -- [array of centroids in 4 dimensions]
    
    Returns:
        [float] -- [distance]
    """

    assert centroids1.shape == centroids2.shape , 'centroids have to have the same dimensions'
    _, distance = sklearn.metrics.pairwise_distances_argmin_min(centroids1, centroids2)
    return distance.sum()


def mds_embedding(centroids):
    # Calculate pairwise distances
    N = len(centroids)
    pos = list(itertools.combinations(range(N), 2))

    # Generate a full pairwise distance matrix, having zeros in the diagonal
    distances = np.zeros(shape=(N, N))
    for p in pos:
        dist = centroid_distance(centroids[p[0]], centroids[p[1]])
        distances[p] = dist
        distances[p[::-1]] = dist

    # Calculate MDS embedding
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(distances)

    coords = results.embedding_
    return coords, distances


def distance_plot(coords, labels=None, colors=None, title=''):
    if labels is None:
        labels = itertools.cycle([''])

    if colors is None:
        cmap = None
        colors = itertools.cycle(['blue'])
    else:
        # Take the colors list and map it to values in range [0, 1]
        ucolors = np.unique(colors)
        cmap = plt.cm.get_cmap('Spectral')
        norm_colors = cmap(np.linspace(0, 1, len(ucolors)))
        color_table = dict(zip(ucolors, norm_colors))
        new_colors = [color_table[x] for x in colors]
        colors = new_colors

    f, ax = plt.subplots(figsize=(10, 10))
    for (x, y), label, color in zip(coords, labels, colors):
        ax.scatter(x, y, c=color, label=label, cmap=cmap)
        ax.annotate(label, (x, y-0.01), color='black', horizontalalignment='center', verticalalignment='top')

    ax.grid(True)

    std = coords.std(axis=0)
    ax.set_title('MDS embedding coord std(%2.2f, %2.2f)\n%s' % (std[0], std[1], title))
    return f


def find_images(paths, grouping=False):
    images = []
    groups = []
    for grp_idx, path in enumerate(paths):
        if os.path.isfile(path):
            images.append(path)
            groups.append(grp_idx)
        else:
            imgs = glob.iglob(os.path.join(path, '**/*.jpg'), recursive=True)
            imgs2 = glob.iglob(os.path.join(path, '**/*.png'), recursive=True)
            imgs = list(imgs)
            imgs2 = list(imgs2)
            images += imgs
            images += imgs2
            groups += [grp_idx]*(len(imgs) + len(imgs2))
    if grouping:
        return tuple(groups), tuple(images)
    else:
        return tuple(images)


def basenames(filenames):
    return [os.path.basename(filename) for filename in filenames]


@img2dist.command()
@click.argument('output_image')
@click.argument('image_path', nargs=-1)
@click.option('--centroids', default=5, type=int, help='Number of centroids to use for calculating image features')
@click.option('--group', is_flag=True, help='Try to color group images by filename similarity')
@click.option('--skip-labels', 'skip_labels', is_flag=True, help='Add no labels to the scatter plot')
def embedding(output_image, image_path, centroids, group, skip_labels):
    """Calculate the distance between two images
    
    Arguments:
        output_image {str} -- [Path to output image]
        image_path {str} -- [Multi paths to image files, or directories]
        centroids {int} -- [number of centroids]
    """
    _embedding(output_image, image_path, centroids, group, skip_labels)

def _embedding(output_images, image_path, centroids, group, skip_labels):
    if group:
        colors, images = find_images(image_path, grouping = True)
    else:
        colors = None
        images = find_images(image_path)
    logging.info('Found %d images ...' % len(images))

    if skip_labels:
        labels = None
    else:
        labels = basenames(images)

    logging.info('Calculating centroids ...')
    centroids = [color_quantization(image, centroids) for image in images]
    coords, distances = mds_embedding(centroids)    

    logging.info(f'Writing figure to {output_images} ...')
    title = 'mean distance %2.2f, max distance %2.2f' % (distances.mean(), distances.max())
    fig = distance_plot(coords, labels=labels, colors=colors, title=title)
    fig.savefig(output_images, dpi=200)


@img2dist.command()
@click.argument('src_img')
@click.argument('dest_img')
@click.option('--centroids', default=5, type=int, help='Number of centroids to use for calculating image features')
def distance(src_img, dest_img, centroids):
    """Calculate the distance between two images
    
    Arguments:
        src_img {str} -- [Path to an image file]
        dest_img {str} -- [Path to an image file]
        centroids {int} -- [number of centroids]
    """
    _distance(src_img, dest_img, centroids)

def _distance(src_img, dest_img, centroids):
    src = color_quantization(src_img, centroids)
    dest = color_quantization(dest_img, centroids)
    print(centroid_distance(src, dest))
    #coords = mds_embedding([c1, c2, c1*0.3, c2*0.9, c1*1.4, c2*1.4])
    #distance_plot(coords, labels='abcdef')

#if __name__ == '__main__':
    """
    c1 = color_quantization(sys.argv[1], int(sys.argv[3]))
    c2 = color_quantization(sys.argv[2], int(sys.argv[3]))
    coords = mds_embedding([c1, c2, c1*0.3, c2*0.9, c1*1.4, c2*1.4])
    distance_plot(coords, labels='abcdef')
    print(centroid_distance(c1, c2))
    """
#    generate_distance_plot_for_image_folder(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    img2dist()