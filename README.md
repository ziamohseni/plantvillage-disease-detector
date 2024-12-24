# Plant Disease Detection Project

This project aims to detect plant diseases from images using a convolutional neural network. It includes training and inference codes, and a Docker-based development environment for easy setup.

## Getting Started

### 1. **Clone the Repository:**

```bash
git clone https://github.com/ziamohseni/plantvillage-disease-detector.git
cd plantvillage-disease-detector
```

### 2. **Download Required Data:**

Download the "data" and "models" folders with their contents from OneDrive:

- Link: [https://kth-my.sharepoint.com/:f:/g/personal/zmohseni_ug_kth_se/EiTxnU7DaaZGhYqGVSgksyUB8d5IOttju_A5wvgazeykRg?e=ahwyLR](https://kth-my.sharepoint.com/:f:/g/personal/zmohseni_ug_kth_se/EiTxnU7DaaZGhYqGVSgksyUB8d5IOttju_A5wvgazeykRg?e=ahwyLR)

After downloading, place both folders in the root directory of the project.

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

### Trained Model

You can find statistics and terminal output for the trained model in `trained-model-stat` folder.
