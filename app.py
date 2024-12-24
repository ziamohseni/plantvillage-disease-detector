import os
import streamlit as st
from src.train import train_model
from src.inference import run_inference
import pandas as pd
import plotly.express as px

st.title("Plant Disease Detection 🌱🩺📊")

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Training Section
st.write("### Training Section")
epochs = st.number_input("Enter the number of epochs:", min_value=1, max_value=100, value=5, step=1)

train_button = st.button("Train Model")
if train_button:
    with st.spinner("Training in progress..."):
        # Train and save the model in the "models/" directory with a timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/plant_disease_model_{timestamp}.keras"

        # Train the model
        history, class_names = train_model(data_dir='data', epochs=epochs, model_path=model_path)

        # Save class names to a file for reuse during inference
        with open("models/class_names.txt", "w") as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")

    st.success(f"Training complete! Model saved as {model_path}")

    # Display training metrics if available
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    if acc and val_acc and loss and val_loss:
        metrics_df = pd.DataFrame({
            'Epoch': range(1, len(acc) + 1),
            'Training Accuracy': acc,
            'Validation Accuracy': val_acc,
            'Training Loss': loss,
            'Validation Loss': val_loss
        })

        st.write("### Training Progress")
        fig_acc = px.line(
            metrics_df, 
            x='Epoch', 
            y=['Training Accuracy', 'Validation Accuracy'], 
            markers=True, 
            title="Accuracy over Epochs"
        )
        fig_loss = px.line(
            metrics_df, 
            x='Epoch', 
            y=['Training Loss', 'Validation Loss'], 
            markers=True, 
            title="Loss over Epochs"
        )

        st.plotly_chart(fig_acc)
        st.plotly_chart(fig_loss)

# Model Selection Section
st.write("### Inference Section")
st.write("Select a trained model for inference:")
model_files = [f for f in os.listdir("models") if f.endswith(".keras")]
if not model_files:
    st.warning("No trained models found. Please train a model first.")
    selected_model = None
else:
    selected_model = st.selectbox("Choose a model:", model_files)

# Load class names from file
if os.path.exists("models/class_names.txt"):
    with open("models/class_names.txt", "r") as f:
        class_names = [line.strip() for line in f]
else:
    class_names = None

# File Uploader for Inference
st.write("Upload an image of a plant leaf to analyze:")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if selected_model:
        # Perform inference
        model_path = os.path.join("models", selected_model)
        inference_results = run_inference(uploaded_file, model_path=model_path, class_names=class_names)
        
        # For backward compatibility: check if the function returns confidence or not
        if len(inference_results) == 3:
            predicted_class, probabilities, confidence = inference_results
        else:
            predicted_class, probabilities = inference_results
            confidence = max(probabilities)

        # Two-column layout: image on the left, diagnosis box on the right
        col1, col2 = st.columns([1, 1])
        with col1:
            # Show the uploaded image
            st.image(uploaded_file, caption="Uploaded Plant Image", use_container_width=True)

        with col2:
            # Green diagnosis box (HTML + inline styling)
            st.markdown(
                f"""
                <div style="
                    background-color: #4CAF50;
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    color: white;
                    ">
                    <h3 style="margin-top: 0;">Diagnosis</h3>
                    <p style="margin-bottom: 0;">
                        <strong>Condition:</strong> {predicted_class}<br/>
                        <strong>Confidence:</strong> {(confidence * 100):.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Display Prediction Probabilities as a Bar Chart below
        st.write("### Prediction Probabilities")
        if class_names:
            df = pd.DataFrame({
                "Class": class_names,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=False)

            fig = px.bar(df, x="Probability", y="Class", orientation="h", title=None)
            st.plotly_chart(fig)
    else:
        st.warning("Please select a trained model to proceed with inference.")
