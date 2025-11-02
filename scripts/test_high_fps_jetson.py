import cv2

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0
):
    return (
        f"v4l2src device=/dev/video0 ! "
        f"image/jpeg, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"videoscale ! "
        f"appsink drop=true"
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Empty frame")
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

