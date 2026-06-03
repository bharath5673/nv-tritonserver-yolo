docker run --gpus all \
  -v /home/bharath/Downloads/test_codes/hcl_task_32/tmp/triton_repo:/models \
  -p 8000:8000 \
  nvcr.io/nvidia/tritonserver:24.09-py3 \
  tritonserver --model-repository=/models



###testing

# curl localhost:8000/v2/health/ready
# curl localhost:8000/v2/health/yolo