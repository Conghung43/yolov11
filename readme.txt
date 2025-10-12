cd /home/conghung/Documents/YOLO

sudo docker run -it --ipc=host --runtime=nvidia \
  -v $(pwd):/workspace \
  ultralytics/ultralytics:latest-jetson-jetpack4


cd /workspace
ls

camera size:
v4l2-ctl --device=/dev/video0 --list-formats-ext

