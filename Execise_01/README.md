# ASIC GAN
In this exercise we have to create a GAN that generates images of melanoma similar
to [isic-archive](https://isic-archive.com).

# ETL of test images
In order to download and preprocess images of the data-set I created a tool
`etl_images.py`.

Try
    python etl_images.py --help


## Download Images
One can download multiple images from the different data sets offered by isic

    python etl_images download --help

Example Download 1000 images to the folder original_images

    python etl_images download ./original_images -n 1000

*Note* Based on the content of the isic-archive fewer pictures might be downloaded
than specified.

## Transform Images
The original images exist in different resolutions and sizes. In order to use them
for training they have to be reshaped to the same size. This is done by center
cropping and resizing.

    python etl_images transform --help

Example: Transform all images in ./original_images to a size of 100x100 and store them in the folder ./test_images

    python etl_images transform ./original_images ./test_images 100 100
