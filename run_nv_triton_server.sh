docker run --gpus all \
  -v tmp/triton_repo:/models \
  -p 8000:8000 \
  bharath5673/nv-tritonserver-yolo:24.09-py3 \
  tritonserver --model-repository=/models



###testing

# curl localhost:8000/v2/health/ready
# curl localhost:8000/v2/health/yolo
