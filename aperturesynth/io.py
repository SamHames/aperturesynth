def save_image(image,filename):
    """ saves the image to the given filename, ensuring reasonable range"""
     # clamp out of range colours
    imageIn = image.copy()
    imageIn[imageIn > 1] = 1
    imageIn[imageIn < 0] = 0
    imageIn *= 255
    im = Image.fromarray(imageIn.astype(numpy.uint8))
    im.save(filename)
