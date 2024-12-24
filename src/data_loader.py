import os
import tensorflow as tf

def get_datasets(data_dir='data', image_size=(256, 256), batch_size=32):
    # Data augmentation pipeline to reduce overfitting
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2)
    ])

    # Create paths to data subdirectories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Load training data and get class names
    raw_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    # Extract class names from directory structure
    class_names = raw_train_dataset.class_names

    # Apply data augmentation to training dataset
    train_dataset = raw_train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Cache dataset in memory and prefetch next batch for better performance
    train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)

    # Load and configure validation dataset (no augmentation needed)
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size
    ).cache().prefetch(tf.data.AUTOTUNE)

    # Load and configure test dataset (no augmentation needed)
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size
    ).cache().prefetch(tf.data.AUTOTUNE)

    # Return all datasets and class names
    return train_dataset, val_dataset, test_dataset, class_names