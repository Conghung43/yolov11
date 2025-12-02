from ultralytics import YOLO
import time
import platform
import numpy as np
import cv2

if platform.system() == "Linux":
    # Load TensorRT engine file
    model = YOLO("model/yolo11n-seg.engine")
else:
    # Load standard YOLO model file
    model = YOLO("model/yolo11n-seg.pt")

# Run inference (image path, URL, or numpy array)
results = model("bus.jpg")

# input: opencv image
# output: objects detect results
def predict_on_image(img, conf):
    results = model(img, conf=conf, imgsz=1280)

    # detection
    # result.boxes.xyxy   # box with xyxy format, (N, 4)
    # cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
    # probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    # boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)

    # # segmentation
    # masks = result.masks.numpy()     # masks, (N, H, W)
    return results

# Test image
# img = "bus.jpg"

# times = []
# for i in range(1000):
#     start = time.time()
#     results = infer_opencv_image(img)
#     end = time.time()

#     inference_time = (end - start) * 1000  # ms
#     times.append(inference_time)
#     print(f"Run {i+1}: {inference_time:.2f} ms")

# avg_time = sum(times) / len(times)
# print(f"\nAverage inference time: {avg_time:.2f} ms")

# # Optional: visualize or save result from last inference
# results[0].save("runs/trt_output.jpg")


