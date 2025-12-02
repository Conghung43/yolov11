import time
import numpy as np
import cv2
from ultralytics import YOLO
import platform

# Initialize model
if platform.system() == "Linux":
    model = YOLO("model/yolo11n.engine")
else:
    model = YOLO("model/yolo11n.pt")

def measure_pipeline_times(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Warm up: Run inference 10 times to stabilize GPU and model performance
    print("Warming up model (10 iterations)...")
    for i in range(10):
        _ = model(img, imgsz=640, conf=0.25, verbose=False)
    print("Warm-up complete. Starting measurements...\n")

    # Pre-processing (CPU): Resize and normalize image
    start_preprocess = time.time()
    img_resized = cv2.resize(img, (640, 640))
    img_normalized = img_resized / 255.0
    end_preprocess = time.time()

    preprocess_time = (end_preprocess - start_preprocess) * 1000  # ms

    # Data Transfer (Host-to-Device): Copy image to GPU
    start_transfer = time.time()
    img_gpu = np.ascontiguousarray(img_normalized, dtype=np.float32)
    end_transfer = time.time()

    transfer_time = (end_transfer - start_transfer) * 1000  # ms

    # Full Inference (GPU + NMS): Run the complete model pipeline
    start_full_inference = time.time()
    results = model(img, imgsz=640, conf=0.25, verbose=False)
    end_full_inference = time.time()

    full_inference_time = (end_full_inference - start_full_inference) * 1000  # ms

    # Detailed breakdown: Model forward pass (without NMS)
    start_model_forward = time.time()
    with_nms_results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
    end_model_forward = time.time()

    # Estimate model forward time by running with very low conf to minimize NMS overhead
    start_forward_only = time.time()
    forward_results = model.predict(img, imgsz=640, conf=0.01, max_det=1, verbose=False)
    end_forward_only = time.time()

    forward_time = (end_forward_only - start_forward_only) * 1000  # ms
    nms_time = full_inference_time - forward_time  # Estimate NMS time

    inference_details = {
        "Model Forward Pass (GPU)": forward_time,
        "NMS + Post-processing": nms_time
    }

    # Post-processing (CPU): Extract results
    start_postprocess = time.time()
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    classes = results[0].boxes.cls.cpu().numpy()  # Class IDs
    end_postprocess = time.time()

    postprocess_time = (end_postprocess - start_postprocess) * 1000  # ms

    return {
        "Pre-processing (CPU)": preprocess_time,
        "Data Transfer (Host-to-Device)": transfer_time,
        "Full Inference (GPU + NMS)": full_inference_time,
        "Post-processing (CPU)": postprocess_time,
        "Inference Details": inference_details
    }

if __name__ == "__main__":
    image_path = "bus.jpg"  # Replace with your test image
    times = measure_pipeline_times(image_path)

    print("Pipeline Timing Analysis:")
    print("=" * 50)
    for stage, time_val in times.items():
        if isinstance(time_val, dict):
            print("\n{}:".format(stage))
            for substage, subtime in time_val.items():
                print("  - {}: {:.2f} ms".format(substage, subtime))
        else:
            print("{}: {:.2f} ms".format(stage, time_val))
    
    print("\n" + "=" * 50)
    total_time = sum([v for v in times.values() if not isinstance(v, dict)])
    print("Total Pipeline Time: {:.2f} ms".format(total_time))