import tensorflow as tf


def build_model(image_size):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 1))
    x = augmentation(inputs)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            activation="relu",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="ChestScan_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
