"""
Synthetic road image generator for demonstration.
Creates realistic-looking road images with potholes.
"""

import cv2
import numpy as np
from typing import List, Tuple
import random


def generate_road_texture(h: int, w: int, dark: bool = False) -> np.ndarray:
    """Generate realistic asphalt texture"""
    base_color = 55 if dark else 80
    road = np.full((h, w, 3), base_color, dtype=np.uint8)
    noise = np.random.randint(0, 25, (h, w, 3), dtype=np.uint8)
    road = cv2.add(road, noise)
    for _ in range(random.randint(30, 80)):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = x1 + random.randint(-100, 100)
        y2 = y1 + random.randint(-5, 5)
        color = random.randint(40, 100)
        cv2.line(road, (x1, y1), (x2, y2), (color, color, color), random.randint(1, 3))
    for _ in range(random.randint(50, 150)):
        x = random.randint(0, w)
        y = random.randint(0, h)
        r = random.randint(1, 4)
        color = random.randint(30, 70)
        cv2.circle(road, (x, y), r, (color, color, color), -1)
    road = cv2.GaussianBlur(road, (3, 3), 0)
    return road


def add_lane_markings(img: np.ndarray) -> np.ndarray:
    """Add lane markings to the road"""
    h, w = img.shape[:2]
    result = img.copy()
    center_x = w // 2
    for y in range(0, h, 60):
        if (y // 60) % 2 == 0:
            cv2.line(result, (center_x, y), (center_x, min(y+40, h)), (180, 180, 100), 3)
    cv2.line(result, (50, 0), (50, h), (220, 220, 150), 4)
    cv2.line(result, (w-50, 0), (w-50, h), (220, 220, 150), 4)
    return result


def add_pothole(img: np.ndarray, x: int, y: int, w: int, h: int, severity: float = 0.5) -> np.ndarray:
    """
    Add a realistic-looking pothole to the image.
    severity: 0.0 (minor) to 1.0 (critical)
    """
    result = img.copy()
    depth_dark = int(20 + 40 * severity)
    base_color = max(10, 55 - depth_dark)
    mask = np.zeros(result.shape[:2], dtype=np.uint8)
    axes = (w // 2, h // 2)
    cv2.ellipse(mask, (x + w//2, y + h//2), axes, 0, 0, 360, 255, -1)
    mask_3d = np.stack([mask, mask, mask], axis=-1) / 255.0
    pothole_region = np.full_like(result, base_color)
    center_noise = np.random.randint(0, 15, result.shape, dtype=np.uint8)
    pothole_region = cv2.subtract(pothole_region, center_noise)
    for _ in range(int(10 * severity)):
        cx = x + random.randint(5, max(6, w-5))
        cy = y + random.randint(5, max(6, h-5))
        cr = random.randint(2, max(3, int(8 * severity)))
        cv2.circle(pothole_region, (cx, cy), cr, (random.randint(5, 30),)*3, -1)
    result = (result * (1 - mask_3d) + pothole_region * mask_3d).astype(np.uint8)
    inner_axes = (max(1, axes[0] - 5), max(1, axes[1] - 5))
    cv2.ellipse(result, (x + w//2, y + h//2), inner_axes, 0, 0, 360, (30, 30, 30), 2)
    edge_color = (100, 90, 80)
    cv2.ellipse(result, (x + w//2, y + h//2), axes, 0, 0, 360, edge_color, 2)
    shadow_offset = max(3, int(5 * severity))
    shadow_mask = np.zeros(result.shape[:2], dtype=np.uint8)
    cv2.ellipse(shadow_mask, (x + w//2 + shadow_offset, y + h//2 + shadow_offset), axes, 0, 0, 360, 60, 3)
    shadow_3d = np.stack([shadow_mask, shadow_mask, shadow_mask], axis=-1) / 255.0
    result = (result * (1 - shadow_3d * 0.4)).astype(np.uint8)
    return result


def generate_sample_image(width: int = 640, height: int = 480, n_potholes: int = 3) -> Tuple[np.ndarray, List[dict]]:
    """Generate a complete sample road image with annotated potholes"""
    road = generate_road_texture(height, width)
    road = add_lane_markings(road)
    ground_truth = []
    placed_boxes = []
    attempts = 0
    while len(placed_boxes) < n_potholes and attempts < 50:
        attempts += 1
        pw = random.randint(60, 180)
        ph = random.randint(40, 120)
        px = random.randint(60, width - pw - 60)
        py = random.randint(60, height - ph - 60)
        overlap = False
        for bx, by, bw, bh in placed_boxes:
            if not (px + pw < bx or bx + bw < px or py + ph < by or by + bh < py):
                overlap = True
                break
        if overlap:
            continue
        severity = random.uniform(0.1, 0.95)
        road = add_pothole(road, px, py, pw, ph, severity)
        placed_boxes.append((px, py, pw, ph))
        sev_label = 'minor' if severity < 0.25 else 'moderate' if severity < 0.50 else 'severe' if severity < 0.75 else 'critical'
        ground_truth.append({'bbox': (px, py, pw, ph), 'severity': sev_label, 'severity_score': severity})
    return road, ground_truth


def generate_video_frames(n_frames: int = 30, width: int = 640, height: int = 480) -> List[np.ndarray]:
    """Generate a sequence of frames simulating driving on a potholed road"""
    frames = []
    base_road = generate_road_texture(height * 3, width)
    base_road = add_lane_markings(base_road)
    potholes = []
    for _ in range(6):
        pw = random.randint(60, 160)
        ph = random.randint(40, 100)
        px = random.randint(60, width - pw - 60)
        py = random.randint(60, height * 2)
        sev = random.uniform(0.2, 0.9)
        base_road = add_pothole(base_road, px, py, pw, ph, sev)
        potholes.append((px, py, pw, ph, sev))
    for i in range(n_frames):
        scroll_y = int(height * 2 * i / n_frames)
        frame = base_road[scroll_y:scroll_y + height, :width].copy()
        if frame.shape[0] < height:
            pad = np.full((height - frame.shape[0], width, 3), 60, dtype=np.uint8)
            frame = np.vstack([frame, pad])
        noise = np.random.randint(0, 8, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        frames.append(frame)
    return frames
