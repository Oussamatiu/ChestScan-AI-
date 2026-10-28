import tensorflow as tf


def preprocess_image(path, label, image_size):
    """Load, resize, and normalize an image file for model training."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def load_image_for_prediction(path, image_size):
    """Prepare a single image using the same steps as the training pipeline."""
    image_bytes = tf.io.read_file(str(path))
    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)
