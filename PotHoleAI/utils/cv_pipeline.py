"""
Classical Computer Vision Pipeline for Pothole Detection
Combines edge detection, contour analysis, and texture features
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


@dataclass
class PotholeDetection:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    severity: str
    severity_score: float
    area_pixels: int
    depth_estimate_cm: float
    texture_score: float
    edge_score: float
    is_duplicate: bool = False
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    frame_id: int = 0
    detection_method: str = "hybrid"


class ClassicalCVPipeline:
    """
    Classical computer vision pipeline implementing:
    - Canny edge detection
    - Contour-based region proposals
    - Texture analysis via LBP and gradient variance
    - Morphological operations for refinement
    """

    def __init__(self):
        self.canny_low = 50
        self.canny_high = 150
        self.min_contour_area = 800
        self.max_contour_area = 80000
        self.blur_kernel = (5, 5)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame: grayscale, denoise, enhance contrast"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        denoised = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return enhanced

    def detect_edges(self, preprocessed: np.ndarray) -> np.ndarray:
        """Multi-scale Canny edge detection"""
        edges_low = cv2.Canny(preprocessed, self.canny_low // 2, self.canny_high // 2)
        edges_high = cv2.Canny(preprocessed, self.canny_low, self.canny_high)
        combined = cv2.bitwise_or(edges_low, edges_high)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(combined, kernel, iterations=1)
        return dilated

    def compute_texture_score(self, region: np.ndarray) -> float:
        """
        Texture analysis using local binary pattern-inspired variance
        Higher variance = more irregular texture = higher pothole probability
        """
        if region.size == 0:
            return 0.0
        grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        texture_var = np.var(region.astype(float))
        grad_var = np.var(gradient_mag)
        score = (texture_var / 2000.0 + grad_var / 5000.0) / 2.0
        return min(float(score), 1.0)

    def find_candidate_regions(self, edges: np.ndarray, preprocessed: np.ndarray) -> List[Tuple]:
        """Find pothole candidate regions using contour detection"""
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_contour_area < area < self.max_contour_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            if not (0.2 < aspect_ratio < 5.0):
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            region = preprocessed[y:y+h, x:x+w]
            texture_score = self.compute_texture_score(region)
            edge_density = np.sum(edges[y:y+h, x:x+w] > 0) / (w * h) if w * h > 0 else 0
            pothole_score = (
                0.30 * min(area / 10000.0, 1.0) +
                0.25 * texture_score +
                0.20 * (1 - solidity) +
                0.15 * min(edge_density * 5, 1.0) +
                0.10 * min(circularity + 0.3, 1.0)
            )
            candidates.append({
                'bbox': (x, y, w, h),
                'contour': contour,
                'area': area,
                'texture_score': texture_score,
                'edge_score': edge_density,
                'pothole_score': pothole_score,
                'solidity': solidity,
            })
        return candidates

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Run full classical CV pipeline on a frame"""
        preprocessed = self.preprocess(frame)
        edges = self.detect_edges(preprocessed)
        candidates = self.find_candidate_regions(edges, preprocessed)
        return preprocessed, edges, candidates


class DeepLearningSimulator:
    """
    Simulates a YOLOv8-style deep learning detector.
    In production, replace this with a real TensorFlow/PyTorch model.
    Uses classical CV features as proxies for DL confidence scores.
    """

    def __init__(self):
        self.confidence_threshold = 0.45
        self.nms_threshold = 0.4

    def predict(self, frame: np.ndarray, cv_candidates: List[dict]) -> List[dict]:
        """
        Simulate deep learning predictions using CV features as proxy.
        Replace this method with actual model.predict() call.
        """
        dl_detections = []
        for candidate in cv_candidates:
            base_score = candidate['pothole_score']
            dl_boost = np.random.normal(0.08, 0.05)
            x, y, w, h = candidate['bbox']
            aspect_noise = abs(0.5 - float(w) / (w + h))
            size_factor = min(candidate['area'] / 15000.0, 1.0)
            dl_confidence = base_score + dl_boost + 0.1 * size_factor - 0.1 * aspect_noise
            dl_confidence = float(np.clip(dl_confidence, 0.0, 0.99))
            if dl_confidence >= self.confidence_threshold:
                dl_detections.append({
                    **candidate,
                    'dl_confidence': dl_confidence,
                    'class': 'pothole',
                })
        dl_detections.sort(key=lambda d: d['dl_confidence'], reverse=True)
        return self._nms(dl_detections)

    def _nms(self, detections: List[dict]) -> List[dict]:
        """Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        kept = []
        suppressed = set()
        for i, det_i in enumerate(detections):
            if i in suppressed:
                continue
            kept.append(det_i)
            for j, det_j in enumerate(detections[i+1:], i+1):
                if j in suppressed:
                    continue
                iou = self._compute_iou(det_i['bbox'], det_j['bbox'])
                if iou > self.nms_threshold:
                    suppressed.add(j)
        return kept

    def _compute_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Compute Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0


class SeverityEstimator:
    """Estimates pothole severity based on size, texture, and depth cues"""

    SEVERITY_LEVELS = {
        'critical': (0.75, 1.00, '#FF1744'),
        'severe':   (0.50, 0.75, '#FF6D00'),
        'moderate': (0.25, 0.50, '#FFD600'),
        'minor':    (0.00, 0.25, '#00E676'),
    }

    def estimate(self, detection: dict, frame_shape: tuple) -> Tuple[str, float, float]:
        """
        Returns (severity_label, severity_score, depth_estimate_cm)
        Severity based on: relative size, texture irregularity, edge sharpness
        """
        frame_area = frame_shape[0] * frame_shape[1]
        x, y, w, h = detection['bbox']
        relative_size = (w * h) / frame_area
        size_score = min(relative_size * 15, 1.0)
        texture_score = detection.get('texture_score', 0.5)
        edge_score = detection.get('edge_score', 0.3)
        severity_score = 0.5 * size_score + 0.3 * texture_score + 0.2 * edge_score
        severity_score = float(np.clip(severity_score, 0.0, 1.0))
        depth_cm = 2.0 + 18.0 * severity_score
        label = 'minor'
        for lname, (lo, hi, _) in self.SEVERITY_LEVELS.items():
            if lo <= severity_score < hi:
                label = lname
                break
        if severity_score >= 0.75:
            label = 'critical'
        return label, round(severity_score, 3), round(depth_cm, 1)


class DuplicateDetector:
    """
    Detects duplicate pothole reports using location + visual similarity.
    Uses perceptual hashing on cropped pothole regions.
    """

    def __init__(self, similarity_threshold: float = 0.85, location_radius_px: int = 80):
        self.similarity_threshold = similarity_threshold
        self.location_radius = location_radius_px
        self.seen_potholes: List[dict] = []
        self.hash_size = 8

    def _phash(self, image: np.ndarray) -> np.ndarray:
        """Compute perceptual hash of image"""
        resized = cv2.resize(image, (self.hash_size * 2, self.hash_size * 2))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        dct = cv2.dct(gray.astype(np.float32))
        dct_low = dct[:self.hash_size, :self.hash_size]
        mean_val = np.mean(dct_low)
        return (dct_low > mean_val).astype(np.uint8)

    def _hamming_similarity(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Compute similarity between two perceptual hashes"""
        total = h1.size
        matches = np.sum(h1 == h2)
        return matches / total

    def _center_distance(self, bbox1: tuple, bbox2: tuple) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

    def is_duplicate(self, frame: np.ndarray, bbox: tuple) -> bool:
        """Check if this detection is a duplicate of a previously seen one"""
        x, y, w, h = bbox
        crop = frame[max(0,y):y+h, max(0,x):x+w]
        if crop.size == 0:
            return False
        current_hash = self._phash(crop)
        for seen in self.seen_potholes:
            dist = self._center_distance(bbox, seen['bbox'])
            if dist < self.location_radius:
                sim = self._hamming_similarity(current_hash, seen['hash'])
                if sim > self.similarity_threshold:
                    return True
        self.seen_potholes.append({'bbox': bbox, 'hash': current_hash})
        if len(self.seen_potholes) > 500:
            self.seen_potholes = self.seen_potholes[-500:]
        return False

    def reset(self):
        self.seen_potholes = []


class GPSTagger:
    """GPS tagging for detected potholes (simulated for demo)"""

    def __init__(self, base_lat: float = 10.7905, base_lon: float = 78.7047):
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.frame_count = 0

    def tag(self, bbox: tuple, frame_shape: tuple) -> Tuple[float, float]:
        """
        Estimate GPS coords based on bounding box position in frame.
        In production: integrate with GPS hardware/API.
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame_shape[:2]
        cx = (x + w/2) / w_frame
        cy = (y + h/2) / h_frame
        lat_offset = (0.5 - cy) * 0.001
        lon_offset = (cx - 0.5) * 0.001
        frame_advance = self.frame_count * 0.00005
        lat = self.base_lat + lat_offset + frame_advance
        lon = self.base_lon + lon_offset
        return round(lat, 6), round(lon, 6)

    def advance_frame(self):
        self.frame_count += 1


class HybridDetector:
    """
    Main hybrid detector combining classical CV + deep learning.
    Orchestrates the full detection pipeline.
    """

    def __init__(self, gps_base: Tuple[float, float] = (10.7905, 78.7047)):
        self.cv_pipeline = ClassicalCVPipeline()
        self.dl_simulator = DeepLearningSimulator()
        self.severity_estimator = SeverityEstimator()
        self.duplicate_detector = DuplicateDetector()
        self.gps_tagger = GPSTagger(*gps_base)
        self.frame_id = 0
        self.total_detections = 0
        self.metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
        }

    def detect(self, frame: np.ndarray) -> Tuple[List[PotholeDetection], dict]:
        """
        Run full hybrid detection pipeline on a single frame.
        Returns detections and debug visualizations.
        """
        t0 = time.time()
        preprocessed, edges, cv_candidates = self.cv_pipeline.process_frame(frame)
        dl_detections = self.dl_simulator.predict(frame, cv_candidates)
        detections = []
        for det in dl_detections:
            bbox = det['bbox']
            severity_label, severity_score, depth_cm = self.severity_estimator.estimate(det, frame.shape)
            is_dup = self.duplicate_detector.is_duplicate(frame, bbox)
            gps_lat, gps_lon = self.gps_tagger.tag(bbox, frame.shape)
            pothole = PotholeDetection(
                bbox=bbox,
                confidence=det['dl_confidence'],
                severity=severity_label,
                severity_score=severity_score,
                area_pixels=det['area'],
                depth_estimate_cm=depth_cm,
                texture_score=det.get('texture_score', 0),
                edge_score=det.get('edge_score', 0),
                is_duplicate=is_dup,
                gps_lat=gps_lat,
                gps_lon=gps_lon,
                frame_id=self.frame_id,
                detection_method="hybrid_cv_dl",
            )
            detections.append(pothole)
        self.gps_tagger.advance_frame()
        proc_time = (time.time() - t0) * 1000
        self.frame_id += 1
        self.total_detections += len([d for d in detections if not d.is_duplicate])
        debug_info = {
            'preprocessed': preprocessed,
            'edges': edges,
            'cv_candidates': len(cv_candidates),
            'dl_detections': len(dl_detections),
            'processing_time_ms': round(proc_time, 2),
            'frame_id': self.frame_id,
        }
        return detections, debug_info

    def visualize(self, frame: np.ndarray, detections: List[PotholeDetection], debug_info: dict) -> np.ndarray:
        """Draw detection results on frame"""
        viz = frame.copy()
        SEVERITY_COLORS = {
            'critical': (0, 23, 255),
            'severe':   (0, 109, 255),
            'moderate': (0, 214, 255),
            'minor':    (0, 230, 118),
        }
        for det in detections:
            x, y, w, h = det.bbox
            color = SEVERITY_COLORS.get(det.severity, (255, 255, 255))
            if det.is_duplicate:
                cv2.rectangle(viz, (x, y), (x+w, y+h), (128, 128, 128), 1)
                cv2.putText(viz, "DUP", (x+2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128), 1)
                continue
            thickness = 3 if det.severity in ('critical', 'severe') else 2
            cv2.rectangle(viz, (x, y), (x+w, y+h), color, thickness)
            overlay = viz.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
            cv2.addWeighted(overlay, 0.15, viz, 0.85, 0, viz)
            label1 = f"{det.severity.upper()} {det.confidence:.0%}"
            label2 = f"~{det.depth_estimate_cm}cm depth"
            label3 = f"GPS: {det.gps_lat:.4f},{det.gps_lon:.4f}"
            label_y = max(y - 35, 15)
            bg_x1, bg_y1 = x, label_y - 14
            bg_x2, bg_y2 = x + 160, label_y + 30
            cv2.rectangle(viz, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
            cv2.rectangle(viz, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
            cv2.putText(viz, label1, (x+3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(viz, label2, (x+3, label_y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(viz, label3, (x+3, label_y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150,200,150), 1, cv2.LINE_AA)
        active = [d for d in detections if not d.is_duplicate]
        info_y = 22
        cv2.rectangle(viz, (0, 0), (260, 100), (15, 15, 15), -1)
        cv2.rectangle(viz, (0, 0), (260, 100), (60, 60, 60), 1)
        cv2.putText(viz, f"Frame #{debug_info['frame_id']}", (8, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(viz, f"Potholes: {len(active)}", (8, info_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)
        cv2.putText(viz, f"CV candidates: {debug_info['cv_candidates']}", (8, info_y+36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        cv2.putText(viz, f"Process: {debug_info['processing_time_ms']:.1f}ms", (8, info_y+54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        fps = 1000 / max(debug_info['processing_time_ms'], 1)
        cv2.putText(viz, f"~{fps:.0f} FPS", (8, info_y+72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,200,255), 1)
        return viz

    def reset_session(self):
        self.duplicate_detector.reset()
        self.frame_id = 0
        self.total_detections = 0
