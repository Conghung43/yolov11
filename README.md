# YOLOv11 Object Detection System

## Overview
This project implements an object detection system using YOLOv8. It includes features for object tracking, metric calculations (e.g., roundness, ripeness), and grading based on thresholds. The system displays boundary-crossing information both in the console and on the video frame.

## Installation

### Prerequisites
- Python 3.8 or higher
- Conda (recommended for environment management)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd yolov11
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create -n yolov11_env python=3.8 -y
   conda activate yolov11_env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Detection System
1. Ensure your YOLO model weights are in the `model/` directory (e.g., `yolo11n.pt`).
2. Run the `main.py` script:
   ```bash
   python scripts/main.py
   ```

### Additional Scripts
- `train.py`: For training the YOLO model.
- `validate_dataset.py`: For validating the dataset.
- `inference.py`: For running inference on images or videos.

## Project Structure
- `data/`: Contains datasets and annotations.
- `model/`: Stores model weights and converted formats.
- `output/`: Stores results and logs.
- `scripts/`: Contains Python scripts for various tasks.

## Contributing
Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License.