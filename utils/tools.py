import tensorflow.compat.v1 as tf

def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_png(string, channels=3)
    image = tf.cast(image, tf.float32)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


def quantize_image(image):
    image = tf.math.floor(image + 0.5)
    image = tf.saturate_cast(image, tf.uint8)
    return image

