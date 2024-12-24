import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

def run_inference(image_file, model_path='plant_disease_model.keras', class_names=None):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the input image
    img = load_img(image_file, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Convert input to a TensorFlow tensor with a fixed dtype and shape
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Run prediction
    predictions = model.predict(img_array, verbose=0)

    # Get the predicted confidence score
    confidence = tf.reduce_max(predictions[0])

    # Get the predicted class name or number based on the class names list
    if class_names:
        predicted_class = class_names[tf.argmax(predictions[0])]
    else:
        predicted_class = f"Class {tf.argmax(predictions[0])}"

    # Return predicted class, prediction vector, and confidence score
    return predicted_class, predictions[0], float(confidence)
