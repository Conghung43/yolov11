import cv2
import math
from typing import Tuple
import numpy as np


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

def is_bbox_inside_area(bbox, detect_area) -> bool:
    """
    Check if a bounding box is completely inside a given area.

    Parameters:
    ----------
    bbox : tuple or list
        Bounding box in format (x_min, y_min, x_max, y_max)
    area : tuple or list
        Area in format (x_min, y_min, x_max, y_max)

    Returns:
    -------
    bool
        True if bbox is fully inside the area, False otherwise
    """

    bx1, by1, bx2, by2 = bbox
    ax1, ay1, ax2, ay2 = detect_area

    # Check if bbox is inside area boundaries
    return (bx1 >= ax1 and by1 >= ay1 and
            bx2 <= ax2 and by2 <= ay2)


def compute_roundness(mask: np.ndarray) -> Tuple[float, Tuple[int, int], float]:
    """
    Compute the roundness of a binary mask and find its best-fit circle.
    Returns: (roundness, center, radius)
    """
    # Ensure binary mask
    mask = (mask > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, (0, 0), 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0, (0, 0), 0.0

    roundness = (4 * math.pi * area) / (perimeter ** 2)

    # Fit minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)

    return roundness, center, radius


def compute_ripeness(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the ripeness of a fruit based on red color percentage in the masked area.
    Uses HSV color space to identify red regions.
    
    Args:
        image: BGR image
        mask: Binary mask where fruit pixels are 1/True
    Returns:
        float: Percentage of red pixels in the masked area (0-100)
    """
    if mask.sum() == 0:  # avoid division by zero
        return 0.0
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define red color range in HSV
    # Red wraps around in HSV, so we need two ranges
    lower_red1 = np.array([0, 70, 50])     # 0-10 in Hue
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])   # 170-180 in Hue
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red regions
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Combine with fruit mask
    fruit_red = cv2.bitwise_and(red_mask, mask * 255)
    
    # Calculate percentages
    total_fruit_pixels = mask.sum()
    red_pixels = (fruit_red > 0).sum()
    
    red_percentage = (red_pixels / total_fruit_pixels) * 100
    
    return red_percentage


def draw_masks_on_frame(frame, result, alpha=0.4, color=(0, 255, 255)):
    """
    Overlay instance masks from an Ultralytics `result` onto `frame` with transparency.

    Supports both mask tensor/data (result.masks.data) and polygon masks
    (result.masks.xy). If no masks available, does nothing.
    """
    if not hasattr(result, 'masks') or result.masks is None:
        return frame

    masks = None
    # Try to extract mask bitmap tensors first
    data = getattr(result.masks, 'data', None)
    if data is not None:
        try:
            # torch Tensor -> numpy
            masks = data.cpu().numpy()
        except Exception:
            try:
                masks = np.array(data)
            except Exception:
                masks = None

    # Fallback: try polygon representation (xy)
    if masks is None:
        polys = getattr(result.masks, 'xy', None)
        if polys is not None:
            h, w = frame.shape[:2]
            masks_list = []
            for poly in polys:
                mask = np.zeros((h, w), dtype=np.uint8)
                # poly can be an array of points or list of arrays
                try:
                    pts = np.array(poly, dtype=np.int32)
                    if pts.ndim == 3:
                        for p in pts:
                            cv2.fillPoly(mask, [p], 255)
                    elif pts.ndim == 2:
                        cv2.fillPoly(mask, [pts], 255)
                    masks_list.append(mask)
                except Exception:
                    continue
            if masks_list:
                masks = np.stack(masks_list, axis=0)

    if masks is None:
        return frame

    # Overlay each instance mask with the same color (can randomize per instance)
    out = frame
    for i in range(masks.shape[0]):
        m = masks[i]
        # m may be float in [0,1] or boolean or 0/255
        if m.dtype != np.uint8:
            m = (m > 0.5).astype(np.uint8) * 255

        mask_bool = m.astype(bool)
        if mask_bool.sum() == 0:
            continue

        colored = np.zeros_like(frame, dtype=np.uint8)
        colored[mask_bool] = color
        # blend
        out = cv2.addWeighted(out, 1.0, colored, alpha, 0)

    return out
