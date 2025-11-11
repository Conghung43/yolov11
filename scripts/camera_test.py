import cv2
import platform
import utils

def main():
    display_count = 0
    status = True
    # Check if OpenCV is built with GStreamer support
    if platform.system() == "Linux":#cv2.getBuildInformation().lower().count('gstreamer'):
        print("Using GStreamer pipeline for video capture.")
        cap = cv2.VideoCapture(utils.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        
    else:
        print("Using default video capture method.")
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
    
if __name__ == "__main__":
    main()