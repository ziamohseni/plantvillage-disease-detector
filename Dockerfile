# We use the official TensorFlow Docker image as the base image for our Docker container
# We choose the GPU version of TensorFlow, as we will be using a GPU to train our model
# Version 2.17.0 of TensorFlow is used, as it is the version that worked with our GPU
FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run Streamlit, binding to 0.0.0.0 and using port 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
