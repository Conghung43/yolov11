import cv2
import numpy as np
import time
import os

def main():
    # Open default camera (0)
    gst = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegdec ! videoconvert ! appsink"
    )

    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create output folder for saving motion frames
    os.makedirs("output", exist_ok=True)

    # Read the first frame as the background
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        return

    # Convert to grayscale and blur
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, gray)

        # Threshold difference
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

        # If motion detected, save image to disk
        # if motion_detected:
        #     timestamp = time.strftime("%Y%m%d-%H%M%S")
        #     output_path = f"output/motion_{timestamp}_{frame_count}.jpg"
        #     cv2.imwrite(output_path, frame)
        #     print(f"[INFO] Motion detected! Saved: {output_path}")

        # Print FPS
        if frame_count % 10 == 0:
            print(f"[INFO] Frame: {frame_count}")

        prev_gray = gray.copy()
        frame_count += 1

        # Sleep a bit to avoid high CPU usage
        time.sleep(0.05)

    cap.release()
    print("Camera released. Exiting...")

if __name__ == "__main__":
    main()

