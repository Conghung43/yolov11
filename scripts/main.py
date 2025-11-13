import cv2
import numpy as np
import time
import os
import platform
import inference
import utils
from collections import deque
import math

# Parameters
min_area = 10000   # smallest object area to keep
max_area = 100000  # largest object area to keep
detect_area = (400, 10, 900, 1050)  # x_min, y_min, x_max, y_max

# Tracker parameters
TRAJECTORY_LENGTH = 30  # how many points to keep for trajectory
MATCH_DISTANCE = 60     # max distance (pixels) to match detection to existing track
MAX_DISAPPEARED = 10    # frames to wait before removing a lost track

# Video output
SAVE_VIDEO = True
OUTPUT_VIDEO = os.path.join("output", "output.mp4")


class SimpleTracker:
    """A very small centroid-based tracker.

    Tracks object centroids across frames. Each track keeps a deque of recent
    centroid points (max length TRAJECTORY_LENGTH). Matching is greedy by
    nearest distance within MATCH_DISTANCE.
    """

    def __init__(self, max_len=TRAJECTORY_LENGTH, max_disappeared=MAX_DISAPPEARED, dist_thresh=MATCH_DISTANCE):
        self.max_len = max_len
        self.max_disappeared = max_disappeared
        self.dist_thresh = dist_thresh
        self.tracks = {}  # id -> {'points': deque, 'disappeared': int}
        self.next_id = 1
        # List to store boundary crossing info for this frame
        self.boundary_crossings = []

    def _new_track(self, point):
        self.tracks[self.next_id] = {
            'points': deque([point], maxlen=self.max_len),
            'disappeared': 0,
            'roundness': [],  # list of roundness values
            'ripeness': [],   # list of ripeness percentages
        }
        self.next_id += 1

    def add_metrics(self, track_id, roundness, ripeness):
        """Add roundness and ripeness metrics to a track."""
        if track_id in self.tracks:
            self.tracks[track_id]['roundness'].append(roundness)
            self.tracks[track_id]['ripeness'].append(ripeness)

    def update_with_metrics(self, detections, metrics=None):
        """Update tracks and add metrics to matched tracks.
        
        Args:
            detections: list of (x, y) centroids
            metrics: list of (roundness, ripeness) tuples matching detections order
        """
        if metrics is None:
            metrics = []
        
        # First, do standard centroid matching
        self.update(detections)
        
        # Now associate metrics with matched tracks
        if len(detections) == 0 or len(metrics) == 0:
            return
        
        # Find which detections were just matched/created
        # We track the last centroid for each track and match it back to detection
        track_ids = list(self.tracks.keys())
        for tid in track_ids:
            last_point = self.tracks[tid]['points'][-1]
            # Find if this track's last point matches a detection
            for di, det in enumerate(detections):
                if di < len(metrics):
                    # Check if this detection is close to the last point (likely the one that was just matched)
                    dist = math.hypot(last_point[0] - det[0], last_point[1] - det[1])
                    # If within a small distance, assume it's this track and add metrics
                    if dist < self.dist_thresh:
                        self.add_metrics(tid, metrics[di][0], metrics[di][1])
                        break

    def update(self, detections):
        """detections: list of (x, y) centroids for this frame"""
        # No detections: increment disappeared counters and remove stale tracks
        if len(detections) == 0:
            remove_ids = []
            for tid, t in self.tracks.items():
                t['disappeared'] += 1
                if t['disappeared'] > self.max_disappeared:
                    remove_ids.append(tid)
            for rid in remove_ids:
                del self.tracks[rid]
            return

        # If no existing tracks, create one for each detection
        if len(self.tracks) == 0:
            for d in detections:
                self._new_track(d)
            return

        # Build list of current last points for tracks
        track_ids = list(self.tracks.keys())
        track_points = [self.tracks[tid]['points'][-1] for tid in track_ids]

        # Keep track of which tracks/detections are matched
        assigned_tracks = set()
        assigned_dets = set()

        # Greedy matching: for each detection find nearest track
        for di, det in enumerate(detections):
            best_tid = None
            best_dist = None
            for ti, tid in enumerate(track_ids):
                if tid in assigned_tracks:
                    continue
                tp = track_points[ti]
                d = math.hypot(tp[0] - det[0], tp[1] - det[1])
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_tid = tid

            if best_dist is not None and best_dist <= self.dist_thresh and best_tid is not None:
                # match
                self.tracks[best_tid]['points'].append(det)
                self.tracks[best_tid]['disappeared'] = 0
                assigned_tracks.add(best_tid)
                assigned_dets.add(di)

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            self._new_track(det)

        # Increment disappeared for unmatched tracks and remove stale ones
        remove = []
        for tid in track_ids:
            if tid in assigned_tracks:
                continue
            self.tracks[tid]['disappeared'] += 1
            if self.tracks[tid]['disappeared'] > self.max_disappeared:
                remove.append(tid)
        for tid in remove:
            if tid in self.tracks:
                del self.tracks[tid]

    def draw(self, frame, detect_area=None):
        """Draw trajectories onto the frame and check for boundary crossings.
        
        Args:
            frame: the frame to draw on
            detect_area: (x_min, y_min, x_max, y_max) to check for left-boundary crossing
        """
        
        for tid, t in self.tracks.items():
            pts = list(t['points'])
            if len(pts) < 2:
                continue
            # color deterministic per id
            color = ((tid * 37) % 255, (tid * 17) % 255, (tid * 97) % 255)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 5)
            # draw last point as a filled circle
            cv2.circle(frame, pts[-1], 10, color, -1)
            
            # Compute and display average metrics
            avg_roundness = np.mean(t['roundness']) if t['roundness'] else 0.0
            avg_ripeness = np.mean(t['ripeness']) if t['ripeness'] else 0.0
            
            # Draw metrics above the last point
            label = f"ID:{tid} R:{avg_roundness:.2f} Ripe:{avg_ripeness:.1f}%"
            cv2.putText(frame, label, (pts[-1][0] - 30, pts[-1][1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Check if object crossed the left boundary (detect_area[0])
            if detect_area is not None and len(pts) >= 2:
                prev_pt = pts[-2]
                curr_pt = pts[-1]
                x_boundary = detect_area[0]
                # Check if crossed from right to left (prev_x >= boundary and curr_x < boundary)
                if prev_pt[0] >= x_boundary and curr_pt[0] < x_boundary:
                    crossing_info = {
                        'tid': tid,
                        'avg_roundness': avg_roundness,
                        'avg_ripeness': avg_ripeness,
                        'count': len(t['roundness']),
                        'roundness_range': (min(t['roundness']), max(t['roundness'])) if t['roundness'] else (0, 0),
                        'ripeness_range': (min(t['ripeness']), max(t['ripeness'])) if t['ripeness'] else (0, 0),
                    }
                    self.boundary_crossings.append(crossing_info)
                    
                    # Also print to console
                    print(f"\n[BOUNDARY CROSSED] Track ID {tid} crossed left boundary (x={x_boundary})")
                    print(f"  Average Roundness: {avg_roundness:.4f}")
                    print(f"  Average Ripeness:  {avg_ripeness:.2f}%")
                    print(f"  Total observations: {len(t['roundness'])}")
                    if t['roundness']:
                        print(f"  Roundness range: {min(t['roundness']):.4f} - {max(t['roundness']):.4f}")
                    if t['ripeness']:
                        print(f"  Ripeness range: {min(t['ripeness']):.2f}% - {max(t['ripeness']):.2f}%")
                    
                    # Determine and print grade
                    if avg_roundness > 0.8 and avg_ripeness > 10:
                        grade = "Grade: 1"
                    elif avg_roundness > 0.8 and avg_ripeness <= 10:
                        grade = "Grade: 2"
                    elif avg_roundness <= 0.8 and avg_ripeness > 10:
                        grade = "Grade: 3"
                    else:
                        grade = "Grade: 4"
                    print(f"  {grade}")
                    print()

    def draw_crossings(self, frame):
        """Draw boundary crossing info onto the frame."""
        if not hasattr(self, 'boundary_crossings') or len(self.boundary_crossings) == 0:
            return frame
        
        y_offset = 30
        crossing = self.boundary_crossings[-1]
        tid = crossing['tid']
        avg_r = crossing['avg_roundness']
        avg_ripe = crossing['avg_ripeness']
        count = crossing['count']
        r_min, r_max = crossing['roundness_range']
        ripe_min, ripe_max = crossing['ripeness_range']
        
        # Determine grade based on thresholds
        if avg_r > 0.8 and avg_ripe > 10:
            grade = "Grade: 1"
        elif avg_r > 0.8 and avg_ripe <= 10:
            grade = "Grade: 2"
        elif avg_r <= 0.8 and avg_ripe > 10:
            grade = "Grade: 3"
        else:
            grade = "Grade: 4"
        
        # Determine color based on track ID
        color = ((tid * 37) % 255, (tid * 17) % 255, (tid * 97) % 255)
        
        # Draw semi-transparent background for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y_offset - 20), (500, y_offset + 110), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Draw text
        cv2.putText(frame, f"[BOUNDARY CROSSED] Track ID: {tid}", 
                    (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Avg Roundness: {avg_r:.4f} (range: {r_min:.4f} - {r_max:.4f})", 
                    (15, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Avg Ripeness: {avg_ripe:.2f}% (range: {ripe_min:.2f}% - {ripe_max:.2f}%)", 
                    (15, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Observations: {count}", 
                    (15, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Determine grade color (green for grade 1, yellow for grade 2, orange for grade 3, red for grade 4)
        if grade == "Grade: 1":
            grade_color = (0, 255, 0)  # Green
        elif grade == "Grade: 2":
            grade_color = (0, 255, 255)  # Yellow
        elif grade == "Grade: 3":
            grade_color = (0, 165, 255)  # Orange
        else:
            grade_color = (0, 0, 255)  # Red
        
        cv2.putText(frame, grade, 
                    (15, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, grade_color, 2)
        
        return frame



def main():
    display_count = 0
    status = True
    # Check if OpenCV is built with GStreamer support
    if platform.system() == "Linux":#cv2.getBuildInformation().lower().count('gstreamer'):
        cap = cv2.VideoCapture(utils.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        
    else:
        # cap = cv2.VideoCapture(0) 
        # Read video from file for testing
        cap = cv2.VideoCapture('images/trim1.mov')  # Change to your camera index if needed
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

    # Initialize simple centroid tracker
    tracker = SimpleTracker()

    # Initialize video writer if requested
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            fps = float(fps)
            if fps <= 0 or math.isnan(fps):
                fps = 30.0
        except Exception:
            fps = 30.0
        frame_size = (prev_frame.shape[1], prev_frame.shape[0])
        os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, frame_size)
        print(f"Saving output video to {OUTPUT_VIDEO} @ {fps} FPS")

    # Save bbox crop images
    bbox_crops = []

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

        # Collect centroids for this frame for tracking
        detections = []
        detection_metrics = []  # store (roundness, ripeness) for each detection

        # Loop through detected contours
        for contour in contours:
            area = cv2.contourArea(contour)
            print(f"Detected contour area: {area}")
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                is_bbox_inside_area_flag = utils.is_bbox_inside_area((x, y, x + w, y + h), detect_area)
                
                # compute centroid from contour moments
                M = cv2.moments(contour)
                if M.get('m00', 0) != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)
                detections.append((cx, cy))

                if is_bbox_inside_area_flag:

                    # predict by YOLOv8
                    results = inference.predict_on_image(frame, conf=0.55)
                    if results[0].masks is None:
                        continue
                    for idx, (mask, box) in enumerate(zip(results[0].masks.data, results[0].boxes.xyxy)):
                        # process if class id = 47 (apple)
                        class_id = int(results[0].boxes.cls[idx].cpu().item())
                        if class_id != 47:
                            continue
                        
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
                        
                        # Store metrics for this detection
                        detection_metrics.append((roundness, ripeness))

                        # Save bbox crop image as a copy (avoid view aliasing so later
                        # drawing on `frame` doesn't modify this saved crop)
                        crop = frame[y1:y2, x1:x2].copy()
                        bbox_crops.append(crop)
                        #remove older crops to limit memory usage
                        if len(bbox_crops)>2:
                            bbox_crops.pop(0)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(frame, f"Area: {int(area)}", (x, y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw the last bbox_crop on the frame for visualization
        if bbox_crops:
            last_crop = bbox_crops[-1]
            crop_h, crop_w = last_crop.shape[:2]
            # Draw last_crop in top right corner and keep scale
            frame[0:crop_h, frame.shape[1]-crop_w:frame.shape[1]] = last_crop

        # Draw checking area
        cv2.rectangle(frame, (detect_area[0], detect_area[1]), 
                      (detect_area[2], detect_area[3]), 
                      [0,255,0], 10)

        # After processing contours, update tracker with metrics and draw trajectories
        tracker.update_with_metrics(detections, detection_metrics)
        tracker.draw(frame, detect_area)
        frame = tracker.draw_crossings(frame)

        # write frame to video if enabled
        if out is not None and out.isOpened():
            # ensure frame is BGR uint8 (should already be)
            out.write(frame)

        cv2.imshow("Frame", frame)
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break




    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Camera released. Exiting...")

if __name__ == "__main__":
    main()
