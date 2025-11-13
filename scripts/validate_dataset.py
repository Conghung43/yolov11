import os
import glob
from pathlib import Path

def validate_dataset(data_dir="data/coco"):
    """Validate COCO dataset structure and consistency"""
    
    issues = []
    
    # Check train set
    train_images = set(Path(p).stem for p in glob.glob(f"{data_dir}/images/train/*"))
    train_labels = set(Path(p).stem for p in glob.glob(f"{data_dir}/labels/train/*.txt"))
    
    print(f"Train images: {len(train_images)}")
    print(f"Train labels: {len(train_labels)}")
    
    # Find mismatches
    images_without_labels = train_images - train_labels
    labels_without_images = train_labels - train_images
    
    if images_without_labels:
        issues.append(f"❌ Images without labels (train): {len(images_without_labels)}")
        for img in list(images_without_labels)[:5]:
            print(f"   - {img}")
    
    if labels_without_images:
        issues.append(f"❌ Labels without images (train): {len(labels_without_images)}")
        for lbl in list(labels_without_images)[:5]:
            print(f"   - {lbl}")
    
    # Check val set
    val_images = set(Path(p).stem for p in glob.glob(f"{data_dir}/images/val/*"))
    val_labels = set(Path(p).stem for p in glob.glob(f"{data_dir}/labels/val/*.txt"))
    
    print(f"\nVal images: {len(val_images)}")
    print(f"Val labels: {len(val_labels)}")
    
    val_images_no_labels = val_images - val_labels
    val_labels_no_images = val_labels - val_images
    
    if val_images_no_labels:
        issues.append(f"❌ Images without labels (val): {len(val_images_no_labels)}")
    if val_labels_no_images:
        issues.append(f"❌ Labels without images (val): {len(val_labels_no_images)}")
    
    # Validate label format
    print("\nValidating label format...")
    bad_labels = []
    for label_path in glob.glob(f"{data_dir}/labels/**/*.txt", recursive=True):
        with open(label_path) as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    bad_labels.append(f"{label_path} line {i+1}: {len(parts)} parts (expected 5)")
    
    if bad_labels:
        issues.append(f"❌ Malformed labels: {len(bad_labels)}")
        for bad in bad_labels[:5]:
            print(f"   - {bad}")
    
    # Summary
    print("\n" + "="*50)
    if issues:
        print("Issues found:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("✅ Dataset validation passed!")
        return True

if __name__ == "__main__":
    validate_dataset()
