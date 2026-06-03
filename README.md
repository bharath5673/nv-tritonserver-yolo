# NVIDIA Triton Inference Server with YOLO Object Detection

A complete setup for running YOLO object detection inference using NVIDIA's Triton Inference Server. This project demonstrates how to deploy YOLO models for real-time object detection with optimized performance using containerized inference infrastructure.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running the Server](#running-the-server)
- [Testing the Inference](#testing-the-inference)
- [Model Details](#model-details)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## 🎯 Overview

This project provides a production-ready setup for deploying YOLO object detection models using **NVIDIA Triton Inference Server**. The implementation includes:

- **NVIDIA Triton Inference Server**: A high-performance inference serving platform supporting multiple backends (TensorRT, ONNX Runtime, etc.)
- **YOLO Detection Model**: Pre-trained YOLO model in ONNX format for object detection
- **HTTP Client**: Python client for making inference requests via REST API
- **Docker Support**: Containerized deployment for easy portability and scaling

### Key Features

- ✅ Real-time object detection using YOLO
- ✅ GPU-accelerated inference with NVIDIA Triton
- ✅ REST API for easy integration
- ✅ COCO dataset object classification (80 classes)
- ✅ Bounding box visualization with confidence scores
- ✅ Original image resolution output
- ✅ JSON-formatted detection results

## 📁 Project Structure

```
NVIDIA-Triton-Inference-Server/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── test_nv_triton.py                 # Inference client script
├── run_nv_triton_server.sh           # Server startup script
├── test.png                          # Sample image for testing
└── tmp/
    └── triton_repo/
        └── yolo/
            ├── config.pbtxt          # Triton model configuration
            └── 1/
                └── model.onnx        # YOLO ONNX model
```

### File Descriptions

| File | Purpose |
|------|---------|
| `test_nv_triton.py` | Python client that connects to the Triton server, sends images for inference, and processes detection results |
| `run_nv_triton_server.sh` | Docker command to launch Triton Inference Server with GPU support |
| `requirements.txt` | Python package dependencies |
| `tmp/triton_repo/yolo/` | Triton model repository containing configuration and model weights |
| `tmp/triton_repo/yolo/config.pbtxt` | Model configuration file specifying input/output specifications |
| `tmp/triton_repo/yolo/1/model.onnx` | YOLO model in ONNX format (version 1) |

## ✅ Prerequisites

### System Requirements

- **NVIDIA GPU**: For GPU-accelerated inference (CUDA compatible)
- **Linux OS**: Ubuntu 18.04+ or similar
- **Docker**: For containerized deployment
- **NVIDIA Docker Runtime**: For GPU access in containers
- **NVIDIA CUDA Toolkit**: For GPU support

### Software Requirements

- Python 3.7+
- Docker with GPU support
- Git

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/bharath5673/nv-tritonserver-yolo.git
cd nv-tritonserver-yolo
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `tritonclient[all]`: Client library for Triton Inference Server (includes HTTP support)

### Step 3: Verify Docker and NVIDIA Docker

```bash
# Check Docker installation
docker --version

# Check NVIDIA Docker runtime
nvidia-docker --version
```

### Step 4: Verify GPU Access

```bash
# Check NVIDIA GPU
nvidia-smi

# Test NVIDIA Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

## 🖥️ Running the Server

### Option 1: Using the Provided Shell Script

```bash
chmod +x run_nv_triton_server.sh
./run_nv_triton_server.sh
```

### Option 2: Manual Docker Command

```bash
docker run --gpus all \
  -v /path/to/tmp/triton_repo:/models \
  -p 8000:8000 \
  nvcr.io/nvidia/tritonserver:24.09-py3 \
  tritonserver --model-repository=/models
```

### Configuration Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--gpus all` | - | Enable all available GPUs |
| `-v /path/to/triton_repo:/models` | - | Mount model repository |
| `-p 8000:8000` | - | Expose HTTP API on port 8000 |
| `24.09-py3` | - | Triton version with Python 3 |
| `--model-repository=/models` | - | Path to models inside container |

### Expected Server Output

```
I0603 12:00:00.000000 1 server.cc:XXX] Started GRPCInferenceService at 0.0.0.0:8001
I0603 12:00:00.000000 1 server.cc:XXX] Started HTTPService at 0.0.0.0:8000
I0603 12:00:00.000000 1 server.cc:XXX] Started Metrics Service at 0.0.0.0:8002
I0603 12:00:00.000000 1 server.cc:XXX] 
I0603 12:00:00.000000 1 server.cc:XXX] NVIDIA Triton Inference Server
...
I0603 12:00:00.000000 1 server.cc:XXX] All models are ready for inferencing
```

### Health Check

```bash
# Check server readiness
curl localhost:8000/v2/health/ready

# Check specific model status
curl localhost:8000/v2/models/yolo
```

## 🧪 Testing the Inference

### Basic Usage

```bash
python test_nv_triton.py
```

### What the Script Does

1. **Connects to Server**: Establishes HTTP connection to `localhost:8000`
2. **Loads Image**: Reads `test.png` from the current directory
3. **Preprocesses**: 
   - Resizes image to 640x640 (model input size)
   - Normalizes pixel values [0-255] → [0-1]
   - Converts BGR → RGB and transposes to CHW format
   - Adds batch dimension
4. **Sends Inference Request**: Sends preprocessed image to YOLO model
5. **Postprocesses Results**:
   - Filters detections by confidence threshold (0.30)
   - Rescales bounding boxes to original image size
   - Extracts class IDs and confidence scores
6. **Visualizes**: Draws bounding boxes with labels on original image
7. **Outputs**: 
   - Prints detection results as JSON
   - Saves annotated image as `output_original_scale.jpg`

### Example Output

```json
[
    {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bbox_xyxy_original": [100, 50, 300, 400]
    },
    {
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.87,
        "bbox_xyxy_original": [350, 150, 600, 350]
    }
]
```

### Using Custom Images

```python
# Replace the default test image
img_path = "path/to/your/image.jpg"
```

## 📦 Model Details

### YOLO Model Specifications

| Specification | Value |
|--------------|-------|
| **Format** | ONNX (Open Neural Network Exchange) |
| **Input Size** | 640x640 pixels |
| **Input Channels** | 3 (RGB) |
| **Output** | Detection tensor with class predictions |
| **Classes** | 80 (COCO dataset classes) |
| **Confidence Threshold** | 0.30 (configurable) |

### COCO Classes (80 total)

```
Objects: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, ...
Animals: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, ...
Sports: sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, ...
Kitchen: bottle, wine glass, cup, fork, knife, spoon, bowl, ...
Furniture: chair, couch, bed, dining table, toilet, ...
Electronics: tv, laptop, mouse, remote, keyboard, cell phone, ...
Appliances: microwave, oven, toaster, sink, refrigerator, ...
Misc: book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```

### Model Configuration (config.pbtxt)

The `config.pbtxt` file defines:
- Model name: "yolo"
- Backend: ONNX Runtime
- Input/output tensor specifications
- Model versioning

## 🔗 API Reference

### HTTP Inference Endpoint

**Endpoint**: `POST http://localhost:8000/v2/models/yolo/infer`

### Request Format

```bash
curl -X POST \
  http://localhost:8000/v2/models/yolo/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "images",
        "shape": [1, 3, 640, 640],
        "datatype": "FP32",
        "data": [...]  # Base64 encoded image data
      }
    ]
  }'
```

### Health Check Endpoints

```bash
# Server health
GET http://localhost:8000/v2/health/ready

# Model status
GET http://localhost:8000/v2/models/yolo
```

## ❌ Troubleshooting

### Issue: "No NVIDIA GPUs detected"

**Solution**:
```bash
# Verify GPU is available
nvidia-smi

# Ensure NVIDIA Docker runtime is installed
docker run --rm --gpus all alpine nvidia-smi
```

### Issue: "Connection refused on localhost:8000"

**Solution**:
- Check if server is running: `curl localhost:8000/v2/health/ready`
- Verify port 8000 is exposed: `docker ps` (look for port mapping)
- Wait 10-15 seconds after starting container for server to fully initialize

### Issue: "Model not loaded" or "Not found"

**Solution**:
```bash
# Verify model repository path is correctly mounted
docker logs <container_id>

# Check config.pbtxt is valid
curl localhost:8000/v2/models/yolo
```

### Issue: "OOM (Out of Memory)" errors

**Solution**:
- Reduce batch size in inference script
- Use smaller model variant
- Allocate more GPU memory in Docker: `--gpus device=0` with memory limits

### Issue: Slow inference

**Solution**:
- Verify GPU usage: `nvidia-smi` (should show high GPU utilization)
- Check if running on CPU instead of GPU
- Enable TensorRT optimization in model config

## 📚 Resources

### Official Documentation
- [NVIDIA Triton Inference Server Documentation](https://github.com/triton-inference-server/server)
- [Triton Python Client Documentation](https://github.com/triton-inference-server/pytriton)
- [YOLO Ultralytics Documentation](https://docs.ultralytics.com/)

### ONNX Format
- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)

### NVIDIA CUDA
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Docker Repository](https://github.com/NVIDIA/nvidia-docker)

### Additional Resources
- [COCO Dataset](https://cocodataset.org/)
- [Object Detection Papers](https://arxiv.org/list/cs.CV/recent)

## 📝 License

This project is provided as-is for educational and research purposes.

## 👤 Author

Bharath

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Last Updated**: June 3, 2026
