import requests
import sys
import os
from PIL import Image
import click
import glob
from concurrent.futures import ProcessPoolExecutor
import itertools
import Augmentor

@click.group()
@click.version_option()
def etl():
    """
    A utility script that can download and transform images from the isic archive.
    https://isic-archive.com
    """

def download_image(imgID, output_path):
    try:
        url = 'https://isic-archive.com/api/v1/image/{}/download'.format(imgID)
        print('Downloading %s ...' % (url))
        r = requests.get(url, allow_redirects=True)
        img = os.path.join(output_path, 'image_%s.jpg' % (imgID)) 
        with open(img, 'wb') as f:
            f.write(r.content)
        return imgID
    except Exception as e:
        print('Downloading %s failed! %s', imgID, e)
        return None

def transform_image(img_src, img_dest, size):
    try:
        print('Transforming %s ...' % img_src)
        I = Image.open(img_src)
        
        # Crop images making them squared, based on the smallest side
        length = min(I.size)
        # Wider than tall
        if I.size[0] > I.size[1]:
            xoffset = int((I.size[0]-length) / 2)
            yoffset = 0
        else:
            xoffset = 0
            yoffset = int((I.size[1]-length) / 2)
        box = (xoffset, yoffset, xoffset+length, yoffset+length)    
        I = I.crop(box)
        I = I.resize(size)

        # Store image
        I.save(img_dest)

        return img_dest
    except Exception as e:
        print('Transforming image %s failed! %s', img_src, e)
        return None



@etl.command()
@click.option('-n', type=int, default=100, help='Number of images')
@click.option('--offset', type=int, default=0, help='Image offset')
@click.option('--dataset', type=str, default='UDA-1', help='Name of the image dataset')
@click.option('--benign_malignant', type=str, default='benign', help='Define image status.')
@click.argument('output_path')
def download(output_path, n, offset, dataset, benign_malignant):
    """
    Download images from isic-archive.com
    """
    # Get images ids
    r = requests.get('https://isic-archive.com/api/v1/image', params={'limit':n, 'offset':offset, 'detail':'true'})
    if r.status_code != 200:
        raise ConnectionError('Could not use ISIC API interface!')

    # Create a list of benign image ids
    meta_list = r.json()
    imgs = []
    for entry in meta_list: 
        # Filter only data from specific dataset
        if entry['dataset']['name'] != dataset:
            continue
        
        # Only take benign
        if entry['meta']['clinical']['benign_malignant'] == benign_malignant:
            i = entry['_id']
            x = entry['meta']['acquisition']['pixelsX']
            y = entry['meta']['acquisition']['pixelsY']
            imgs.append(i)
    
    print('Found %d/%d images matching the search requirements ...' % (len(imgs), n))

    # Download images
    os.makedirs(output_path, exist_ok=True)

    with ProcessPoolExecutor(max_workers=None) as executor:
        nImages = [1 for x in executor.map(download_image, imgs, itertools.repeat(output_path)) if x is not None]
        
    print('Finished download %d images!' %(sum(nImages)))


@etl.command()
@click.argument('input_path', type=str)
@click.argument('output_path', type=str)
@click.argument('x', type=int)
@click.argument('y', type=int)
def transform(input_path, output_path, x, y):
    """
    crop/resize images to given size
    """
    os.makedirs(output_path, exist_ok=True)
    # Transform pictures
    size = (x, y)
    img_src = glob.glob(os.path.join(input_path, '*.jpg')) 
    img_dest = [os.path.join(output_path, os.path.basename(img)) for img in img_src]

    with ProcessPoolExecutor(max_workers=None) as executor:
        executor.map(transform_image, img_src, img_dest, itertools.repeat(size))
        nImages = [1 for x in executor.map(transform_image, img_src, img_dest, itertools.repeat(size)) if x is not None]

    print('Finished transforming %d images!' %(sum(nImages)))


@etl.command()
@click.argument('input_path', type=str)
@click.argument('n', type=int)
def augment(input_path, n):
    """
    Apply a set of augmentation operations
    """
    p = Augmentor.Pipeline(input_path)
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
#p.ground_truth("/path/to/ground_truth_images")
# Add operations to the pipeline as normal:
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)
    p.sample(n)


if __name__ == '__main__':
    etl()
