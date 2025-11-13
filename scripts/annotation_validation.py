import glob

for label_path in glob.glob("data/coco/labels/**/*.txt", recursive=True):
    with open(label_path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Bad label in {label_path} line {i+1}: {line}")