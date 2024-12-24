# Plant Disease Detection Project

This project aims to detect plant diseases from images using a convolutional neural network. It includes training and inference codes, and a Docker-based development environment for easy setup.

## Prerequisites

- **Docker Desktop:** Ensure you have Docker Desktop installed and running.

---

## Running the Project

If you want to run the model:

### 1. **Build the Docker Image:**

```bash
docker build -t plant-disease:latest .
```

This command uses the `Dockerfile` to create an image with all dependencies installed.

### 2. **Run the Container:**

```bash
docker run --rm -p 8501:8501 plant-disease:latest
```

Or run with GPU support (if you have Nvidia GPU)

```bash
docker run --rm -p 8501:8501 --gpus all plant-disease:latest
```

**Stopping the Container:**  
Press `Ctrl + C` in the terminal to stop the container.

## Notes

### Dependencies:

If you need to add new Python packages, update `requirements.txt` and rebuild the image by following step 1.
