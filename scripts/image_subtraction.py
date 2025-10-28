import cv2
import numpy as np
import time
import os
import platform

# Parameters
min_area = 4000   # smallest object area to keep
max_area = 10000  # largest object area to keep

def gstreamer_pipeline(
    capture_width=800,
    capture_height=600,
    display_width=800,
    display_height=600,
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



def main():
    # Check if OpenCV is built with GStreamer support
    if platform.system() == "Linux":#cv2.getBuildInformation().lower().count('gstreamer'):
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        
    else:
        # cap = cv2.VideoCapture(1) 
        # Read video from file for testing
        cap = cv2.VideoCapture('/Users/nguyenconghung/Documents/Video/ConveyerBelt/conveyor.mov')  # Change to your camera index if needed
        # Set manual exposure (values depend on your camera)
        # Usually negative values for webcams = auto-exposure off
        #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 disables auto mode (depends on backend)
        # cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # smaller = darker image

        # Try setting ISO (not supported by most webcams)
        # cap.set(cv2.CAP_PROP_ISO_SPEED, 400)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    os.makedirs("output", exist_ok=True)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    frame_count = 0
    prev_time = time.time()
    
    # Create KNN background subtractor
    fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)

    # Structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fgmask = fgbg.apply(frame)

        # Morphological filtering to clean noise
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Optional: dilate slightly to fill gaps
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        # Find contours from the foreground mask
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through detected contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {int(area)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display results
        cv2.imshow("Foreground Mask", fgmask)
        cv2.imshow("Detected Objects", frame)

        # Exit on ESC key
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Exiting...")

if __name__ == "__main__":
    main()
