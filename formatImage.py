#!python3
"""
formatImage.py
"""
import PIL
import matplotlib.pyplot as plt
import codeTime as cT
import numpy as np

def crop_background(img):
    # Create image with the size of input image
    # and pixel value of first top left pixel
    bg = PIL.Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if not bbox:
        return img
    return img.crop(bbox)


def format_img_mnist(img):
    img = crop_background(img)
    width = float(img.size[0])
    height = float(img.size[1])
    newimg = PIL.Image.new('L', (28, 28), 255)

    if width > height:
        nheight = int(round(20.0 / width * height, 0))
        if nheight == 0:
            nheight = 1
        nwidth = 20
        hpos = int(round((28 - nheight) / 2, 0))
        vpos = 4
    else:
        nheight = 20
        nwidth = int(round(20.0 / height * width, 0))
        if nwidth == 0:
            nwidth = 1
        hpos = 4
        vpos = int(round((28 - nwidth) / 2, 0))

    img = img.resize((nwidth, nheight), PIL.Image.ANTIALIAS)
    newimg.paste(img, (vpos, hpos))
    return newimg


@cT.exeTime
def format_img_mnist_data(img):
    newimg = format_img_mnist(img)
    imgdata = np.array(newimg, dtype=np.dtype(np.float32)).reshape(1, 784)
    imgdata = (255 - imgdata) / 255

    return imgdata


if __name__ == "__main__":
    path = "hand/7test.png"
    # Open image in black and white mode
    img = PIL.Image.open(path).convert('L')

    # !!! Format custom image to comply with mnist data !!!
    cimg = crop_background(img)  # crop image

    # Resize and rescale image for mnist
    nimg = format_img_mnist(cimg)

    # Format image as data
    imgdata = format_img_mnist_data(nimg)

    # # Pillow show image
    # im.show()
    # newimg.show()

    # matplotlib show image
    plt.imshow(img, cmap='gray')
    plt.figure()
    plt.imshow(cimg, cmap='gray')
    plt.figure()
    plt.imshow(nimg, cmap='gray')
    plt.show()
