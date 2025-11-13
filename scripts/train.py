from ultralytics import YOLO

# Load a model
model = YOLO("model/yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model and store results
results = model.train(
    data="data/coco/coco.yaml",
    epochs=2,
    imgsz=640,
    device="cpu",  # Use CPU instead of MPS due to PyTorch MPS backend issues
    batch=32,
    patience=20
)

# Get mAP 50-95 and other metrics
print(f"mAP 50-95: {results.results_dict['metrics/mAP50-95(B)']}")
print(f"mAP 50: {results.results_dict['metrics/mAP50(B)']}")
print(f"mAP 75: {results.results_dict['metrics/mAP75(B)']}")

# Or access all metrics
print("\nAll Metrics:")
for metric, value in results.results_dict['metrics'].items():
    print(f"{metric}: {value}")