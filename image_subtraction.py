import cv2
import numpy as np
import time
import os

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
    if cv2.getBuildInformation().lower().count('gstreamer'):
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        
    else:
        cap = cv2.VideoCapture(0)    

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # === FPS Calculation ===
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        print(f"FPS: {fps:.2f} {frame.shape[0]}x{frame.shape[1]} ")

        # Crop

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        prev_gray = gray.copy()
        frame_count += 1

        # Optional: Display frame (press 'q' to exit)
        cv2.imshow("Motion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sleep to reduce CPU usage
        # time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Exiting...")

if __name__ == "__main__":
    main()
