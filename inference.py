from ultralytics import YOLO
import time

# Load TensorRT engine file
model = YOLO("model/yolo11n-seg.engine")

# Run inference (image path, URL, or numpy array)
results = model("bus.jpg")

# Test image
img = "bus.jpg"

times = []
for i in range(1000):
    start = time.time()
    results = model(img)
    end = time.time()

    inference_time = (end - start) * 1000  # ms
    times.append(inference_time)
    print(f"Run {i+1}: {inference_time:.2f} ms")

avg_time = sum(times) / len(times)
print(f"\nAverage inference time: {avg_time:.2f} ms")

# Optional: visualize or save result from last inference
results[0].save("runs/trt_output.jpg")
