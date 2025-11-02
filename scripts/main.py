import cv2
import numpy as np
import time
import os
import platform
import inference
import utils

# Parameters
min_area = 4000   # smallest object area to keep
max_area = 10000  # largest object area to keep
detect_area = (630, 410, 770, 620)  # x_min, y_min, x_max, y_max


def main():
    display_count = 0
    status = True
    # Check if OpenCV is built with GStreamer support
    if platform.system() == "Linux":#cv2.getBuildInformation().lower().count('gstreamer'):
        cap = cv2.VideoCapture(utils.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        
    else:
        # cap = cv2.VideoCapture(1) 
        # Read video from file for testing
        cap = cv2.VideoCapture('Conveyor.mov')  # Change to your camera index if needed
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
                is_bbox_inside_area_flag = utils.is_bbox_inside_area((x, y, x + w, y + h), detect_area)
                if is_bbox_inside_area_flag:
                    frame = cv2.imread("images/3.jpg")

                    # predict by YOLOv8
                    results = inference.predict_on_image(frame, conf=0.55)

                    for mask, box in zip(results[0].masks.data, results[0].boxes.xyxy):
                        # Convert mask tensor to numpy (uint8)
                        mask = mask.cpu().numpy()
                        mask = (mask > 0.5).astype(np.uint8)

                        # Resize mask to image shape if needed
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                        # Create colored overlay for mask
                        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                        colored_mask = np.zeros_like(frame, dtype=np.uint8)
                        for c in range(3):
                            colored_mask[:, :, c] = mask * color[c]

                        # Blend mask with original image
                        # frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

                        # Optionally draw bounding box
                        x1, y1, x2, y2 = box.int().cpu().numpy()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)


                        # Given mask, Find the roundness of the mask
                        # Given the foreground mask, compute roundness for the detected bbox
                        # Use the foreground mask (fgmask) cropped to the bbox as the input
                        # Compute roundness and circle
                        roundness, center, radius = utils.compute_roundness(mask)

                        # Draw circle and roundness value
                        cv2.circle(frame, center, radius, (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"R={roundness:.2f}",
                            (center[0] - 30, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        
                        # Calculate and display ripeness percentage
                        ripeness = utils.compute_ripeness(frame, mask)
                        cv2.putText(
                            frame,
                            f"Ripe: {ripeness:.1f}%",
                            (center[0] - 40, center[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                    cv2.imwrite(f"output/frame.jpg", frame)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(frame, f"Area: {int(area)}", (x, y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        # Exit on ESC key
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Exiting...")

if __name__ == "__main__":
    main()
