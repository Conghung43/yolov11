# from ultralytics import YOLO

# # 1. Load your model
# model = YOLO("..model/yolo11n-seg.pt")  # or "best.pt" if you trained your own

# # 2. Validate on your dataset
# results = model.val(
#     data="coco.yaml",  # or "coco.yaml"
#     imgsz=640,
#     save_json=True,     # saves COCO-format results for per-class mAP
#     verbose=True
# )

# # 3. Print per-class mAP
# names = results.names
# maps = results.box.ap_class  # average precision per class

# print("\n📊 Per-class mAP:")
# for i, ap in enumerate(maps):
#     print(f"{names[i]:15s}: {ap:.4f}")

# # 4. Optionally save to a text file
# with open("map_per_class.txt", "w") as f:
#     for i, ap in enumerate(maps):
#         f.write(f"{names[i]:15s}: {ap:.4f}\n")

# print("\n✅ Per-class mAP saved to map_per_class.txt")


from ultralytics import YOLO

# Load your pre-trained model
model = YOLO('model/yolo11n-seg.pt') 

# Run evaluation on the test set using your coco.yaml file
metrics = model.val(data='coco.yaml', split='test') 

# Print key metrics, such as mAP
print(f"Mean Average Precision (mAP) is: {metrics.box.map}")
