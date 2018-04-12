# ASIC GAN
In this exercise we have to create a GAN that generates images of melanoma similar
to [isic-archive](https://isic-archive.com).

# ETL of test images
In order to download and preprocess images of the data-set I created a tool
`isic_dataset.py`.

Try
    python isic_dataset.py --help


## Download Images
One can download multiple images from the different data sets offered by isic

    python isic_dataset.py download --help

Example Download 1000 images to the folder original_images

    python isic_dataset.py download ./original_images -n 1000

*Note* Based on the content of the isic-archive fewer pictures might be downloaded
than specified.

## Transform Images
The original images exist in different resolutions and sizes. In order to use them
for training they have to be reshaped to the same size. This is done by center
cropping and resizing.

    python isic_dataset.py transform --help

Example: Transform all images in ./original_images to a size of 100x100 and store them in the folder ./test_images

    python isic_dataset.py transform ./original_images ./test_images 100 100
