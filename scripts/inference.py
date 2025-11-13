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

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

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


