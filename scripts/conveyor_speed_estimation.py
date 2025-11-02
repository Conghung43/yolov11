import cv2
import numpy as np
import time
from scipy.stats import gaussian_kde

class ConveyorSpeedTracker:
    def __init__(self, video_source=0, frame_rate=10, buffer_size=100):
        self.cap = cv2.VideoCapture(video_source)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
        self.frame_rate = frame_rate
        self.buffer_size = buffer_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.prev_rect = None
        self.prev_time = 0
        self.speeds = []
        self.conveyor_speed = None
        self.is_tracking = False

    def is_image_updated(self, frame, prev_frame):
        if prev_frame is None:
            return True
        diff = cv2.absdiff(frame, prev_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(gray) > 0

    def get_center(self, rect):
        x, y, w, h = rect
        return (x + w / 2, y + h / 2)

    def is_valid_movement(self, dx, rect, prev_rect, frame_width):
        w1, h1 = rect[2], rect[3]
        w2, h2 = prev_rect[2], prev_rect[3]
        width_ratio = min(w1, w2) / max(w1, w2)
        height_ratio = min(h1, h2) / max(h1, h2)
        return (
            width_ratio > 0.8 and height_ratio > 0.8 and
            20 < rect[0] < frame_width - rect[2] - 20 and
            abs(dx) > 0
        )

    def estimate_mode(self, data, bandwidth=None):
        if len(data) == 0:
            return None
        kde = gaussian_kde(data, bw_method=bandwidth)
        xs = np.linspace(min(data), max(data), 1000)
        ys = kde(xs)
        return xs[np.argmax(ys)]

    def process(self):
        prev_frame = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            if not self.is_image_updated(frame, prev_frame):
                time.sleep(1 / self.frame_rate)
                continue
            prev_frame = frame.copy()

            fgmask = self.fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                self.is_tracking = False
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # find largest contour
            largest = max(contours, key=cv2.contourArea)
            rect = cv2.boundingRect(largest)

            if self.prev_rect is not None:
                c1 = self.get_center(rect)
                c2 = self.get_center(self.prev_rect)
                dx = c1[0] - c2[0]
                frame_width = frame.shape[1]

                if self.is_valid_movement(dx, rect, self.prev_rect, frame_width):
                    now = time.time()
                    if self.prev_time != 0:
                        dt = now - self.prev_time
                        if dt > 0:
                            speed = dx / dt
                            self.speeds.append(speed)
                            print(f"dx={dx:.2f}, dt={dt:.4f}, speed={speed:.2f}")

                            if len(self.speeds) >= self.buffer_size:
                                self.conveyor_speed = self.estimate_mode(self.speeds, bandwidth=frame_width / 100)
                                print(f"\n🟢 Estimated Conveyor Speed: {self.conveyor_speed:.2f} pixels/sec")
                                self.speeds.clear()
                    self.prev_time = now
                    self.is_tracking = True
                else:
                    self.is_tracking = False
            else:
                self.is_tracking = False

            self.prev_rect = rect

            # draw result
            x, y, w, h = rect
            color = (0, 255, 0) if self.is_tracking else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.imshow("Tracking", frame)

            if cv2.waitKey(int(1000 / self.frame_rate)) == 27:  # ESC to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ConveyorSpeedTracker(video_source='Conveyor.mov', frame_rate=10)
    tracker.process()