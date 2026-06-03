import numpy as np
import cv2
import json
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

# COCO class names
coco_classes = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

client = httpclient.InferenceServerClient("localhost:8000")

img_path = "test.png"
img = cv2.imread(img_path)
orig_h, orig_w = img.shape[:2]

# Resize for model
img_resized = cv2.resize(img, (640, 640))
img_input = img_resized.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)

# Triton input
inputs = []
inputs.append(
    httpclient.InferInput("images", img_input.shape, np_to_triton_dtype(img_input.dtype))
)
inputs[0].set_data_from_numpy(img_input)

outputs = []
outputs.append(httpclient.InferRequestedOutput("output0"))

response = client.infer(model_name="yolo", inputs=inputs, outputs=outputs)
result = response.as_numpy("output0")[0]

# Scaling factors
scale_x = orig_w / 640
scale_y = orig_h / 640

detections_list = []

for det in result:
    x1, y1, x2, y2, conf, cls_id = det

    if conf > 0.30:
        cls_id = int(cls_id)
        class_name = coco_classes[cls_id]

        # 🔥 Rescale to original image size
        x1_orig = int(x1 * scale_x)
        y1_orig = int(y1 * scale_y)
        x2_orig = int(x2 * scale_x)
        y2_orig = int(y2 * scale_y)

        detection = {
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": float(conf),
            "bbox_xyxy_original": [
                x1_orig, y1_orig, x2_orig, y2_orig
            ]
        }

        detections_list.append(detection)

        # Draw on ORIGINAL image
        cv2.rectangle(
            img,
            (x1_orig, y1_orig),
            (x2_orig, y2_orig),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"{class_name} {conf:.2f}",
            (x1_orig, y1_orig - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

print(json.dumps(detections_list, indent=4))

cv2.imwrite("output_original_scale.jpg", img)
print("\nSaved output as output_original_scale.jpg")