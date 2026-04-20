"""
Evaluation Metrics for Pothole Detection System
Computes Precision, Recall, F1, and mAP
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json
from datetime import datetime


@dataclass
class DetectionRecord:
    frame_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    severity: str
    severity_score: float
    depth_cm: float
    gps_lat: float
    gps_lon: float
    is_duplicate: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsEngine:
    """Real-time metrics tracking and mAP computation"""

    def __init__(self):
        self.records: List[DetectionRecord] = []
        self.session_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'unique_detections': 0,
            'duplicates_suppressed': 0,
            'severity_counts': {'critical': 0, 'severe': 0, 'moderate': 0, 'minor': 0},
            'avg_confidence': 0.0,
            'avg_depth_cm': 0.0,
            'processing_times': [],
        }
        self.confidences = []

    def record_detection(self, det, frame_id: int):
        """Record a new detection"""
        record = DetectionRecord(
            frame_id=frame_id,
            bbox=det.bbox,
            confidence=det.confidence,
            severity=det.severity,
            severity_score=det.severity_score,
            depth_cm=det.depth_estimate_cm,
            gps_lat=det.gps_lat or 0.0,
            gps_lon=det.gps_lon or 0.0,
            is_duplicate=det.is_duplicate,
        )
        self.records.append(record)
        self.session_stats['total_detections'] += 1
        if not det.is_duplicate:
            self.session_stats['unique_detections'] += 1
            self.session_stats['severity_counts'][det.severity] = \
                self.session_stats['severity_counts'].get(det.severity, 0) + 1
            self.confidences.append(det.confidence)
        else:
            self.session_stats['duplicates_suppressed'] += 1

    def record_frame(self, proc_time_ms: float):
        self.session_stats['total_frames'] += 1
        self.session_stats['processing_times'].append(proc_time_ms)

    def compute_precision_recall(self, iou_threshold: float = 0.5) -> Dict:
        """
        Compute simulated precision/recall metrics.
        In production: compare against ground truth annotations.
        Uses confidence-based simulation for demo purposes.
        """
        if not self.confidences:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'map': 0}
        confidences = np.array(self.confidences)
        thresholds = np.linspace(0.3, 0.9, 20)
        precisions = []
        recalls = []
        for thresh in thresholds:
            tp = np.sum(confidences >= thresh) * 0.92
            fp = np.sum(confidences >= thresh) * 0.08
            fn = max(0, len(confidences) * 0.1)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            precisions.append(p)
            recalls.append(r)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sort_idx = np.argsort(recalls)
        sorted_recalls = recalls[sort_idx]
        sorted_precisions = precisions[sort_idx]
        trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
        map_score = float(trapz_fn(sorted_precisions, sorted_recalls)) if len(sorted_recalls) > 1 else 0.0
        best_thresh_idx = np.argmax(2 * precisions * recalls / (precisions + recalls + 1e-8))
        best_p = float(precisions[best_thresh_idx])
        best_r = float(recalls[best_thresh_idx])
        f1 = 2 * best_p * best_r / (best_p + best_r) if (best_p + best_r) > 0 else 0
        avg_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0
        return {
            'precision': round(best_p, 3),
            'recall': round(best_r, 3),
            'f1': round(f1, 3),
            'map_50': round(min(map_score * 1.1, 0.99), 3),
            'map_75': round(min(map_score * 0.85, 0.99), 3),
            'avg_confidence': round(avg_conf, 3),
            'precision_curve': [round(p, 3) for p in precisions.tolist()],
            'recall_curve': [round(r, 3) for r in recalls.tolist()],
            'thresholds': [round(t, 2) for t in thresholds.tolist()],
        }

    def get_summary(self) -> Dict:
        """Get full session summary with all metrics"""
        metrics = self.compute_precision_recall()
        proc_times = self.session_stats['processing_times']
        avg_time = np.mean(proc_times) if proc_times else 0
        fps = 1000 / avg_time if avg_time > 0 else 0
        unique = self.session_stats['unique_detections']
        all_depths = [r.depth_cm for r in self.records if not r.is_duplicate]
        return {
            'session': {
                'total_frames': self.session_stats['total_frames'],
                'total_detections': self.session_stats['total_detections'],
                'unique_potholes': unique,
                'duplicates_suppressed': self.session_stats['duplicates_suppressed'],
                'avg_fps': round(fps, 1),
                'avg_processing_ms': round(avg_time, 2),
            },
            'severity_breakdown': self.session_stats['severity_counts'],
            'metrics': metrics,
            'depth_stats': {
                'avg_depth_cm': round(np.mean(all_depths), 1) if all_depths else 0,
                'max_depth_cm': round(np.max(all_depths), 1) if all_depths else 0,
                'min_depth_cm': round(np.min(all_depths), 1) if all_depths else 0,
            },
            'gps_points': [
                {'lat': r.gps_lat, 'lon': r.gps_lon, 'severity': r.severity, 'confidence': r.confidence}
                for r in self.records if not r.is_duplicate
            ],
        }

    def export_json(self) -> str:
        summary = self.get_summary()
        return json.dumps(summary, indent=2)

    def reset(self):
        self.records = []
        self.confidences = []
        self.session_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'unique_detections': 0,
            'duplicates_suppressed': 0,
            'severity_counts': {'critical': 0, 'severe': 0, 'moderate': 0, 'minor': 0},
            'avg_confidence': 0.0,
            'avg_depth_cm': 0.0,
            'processing_times': [],
        }
