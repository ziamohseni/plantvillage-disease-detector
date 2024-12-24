from src.data_loader import get_datasets
from src.model import build_model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf

def train_model(data_dir='data', epochs=20, model_path='models/plant_disease_model.keras'):
    # Check if GPU is available and which one is being used by TensorFlow
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))
    
    # Load datasets and class names from the data directory
    train_dataset, val_dataset, test_dataset, class_names = get_datasets(data_dir)
    
    # Fixed class weight computation
    y_train = []
    for _, y in train_dataset:
        y_train.extend(y.numpy())
    y_train = np.array(y_train)
    
    # Compute class weights for imbalanced datasets
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    # Build and train the model using the datasets
    model = build_model(input_shape=(256, 256, 3), num_classes=len(class_names))
    
    # Define callbacks for early stopping, learning rate reduction, and model checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train the model and save it to disk
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Save the model to disk
    model.save(model_path)
    
    # Evaluate the model on the test dataset and print results to the console
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    return history, class_names