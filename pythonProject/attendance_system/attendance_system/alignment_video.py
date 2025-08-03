import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from typing import Tuple, List, Dict, Any, Optional
import time
import uuid
from collections import defaultdict
from scipy.spatial.distance import cosine
import logging
import pickle
import warnings
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("face_tracker")

# Constants
EXPANSION_FACTOR = 1.3  # Increased to capture more context
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8
MAX_FRAMES_TO_SKIP = 30
IOU_THRESHOLD = 0.25  # Reduced to be more lenient in matching
FEATURE_MATCH_THRESHOLD = 0.7  # Threshold for feature similarity matching
MEMORY_SIZE = 10  # Store this many previous embeddings for each track
MAX_DET_SIZE = (1280, 1280)  # Maximum detection size for better accuracy

# Initialize the face analyzer with advanced settings
class FaceAnalyzer:
    def __init__(self, det_size=(640, 640), ctx_id=0, name='buffalo_l'):
        self.app = FaceAnalysis(name=name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.det_sizes = [(640, 640), (320, 320), (1280, 1280)]
        
    def get_faces(self, frame, min_confidence=DEFAULT_CONFIDENCE_THRESHOLD):
        """Try different detection sizes to optimize face detection"""
        best_faces = []
        best_size = None
        
        # Start with the default size
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        best_faces = self.app.get(frame)
        best_faces = [f for f in best_faces if hasattr(f, 'det_score') and f.det_score >= min_confidence]
        
        if len(best_faces) == 0:
            # Try other sizes if no faces detected
            for det_size in self.det_sizes:
                self.app.prepare(ctx_id=0, det_size=det_size)
                faces = self.app.get(frame)
                faces = [f for f in faces if hasattr(f, 'det_score') and f.det_score >= min_confidence]
                
                if len(faces) > len(best_faces):
                    best_faces = faces
                    best_size = det_size
        
        return best_faces

class FaceTrack:
    """Class representing a single face track with temporal information"""
    def __init__(self, track_id, bbox, embedding=None, frame_num=0):
        self.id = track_id
        self.bbox = bbox
        self.last_seen = frame_num
        self.first_seen = frame_num
        self.embeddings = []  # Store multiple embeddings for better matching
        self.appearances = 0  # Count how many times this face appeared
        self.skip_count = 0
        self.best_quality_embedding = None
        self.best_quality_score = 0
        self.best_face_image = None
        self.best_face_frame = None
        self.gone_off_screen = False
        self.location_history = []  # Track location history
        
        if embedding is not None:
            self.add_embedding(embedding, 0)
    
    def add_embedding(self, embedding, quality_score):
        """Add a new embedding to the track and update best quality if better"""
        self.embeddings.append(embedding)
        # Keep only the most recent embeddings
        if len(self.embeddings) > MEMORY_SIZE:
            self.embeddings.pop(0)
        
        # Update best quality embedding if this one is better
        if quality_score > self.best_quality_score:
            self.best_quality_embedding = embedding
            self.best_quality_score = quality_score
    
    def get_average_embedding(self):
        """Get the average embedding from all stored embeddings"""
        if not self.embeddings:
            return None
        avg_embedding = np.mean(self.embeddings, axis=0)
        return avg_embedding / np.linalg.norm(avg_embedding)  # Normalize
    
    def update_location(self, bbox, frame_num):
        """Update track location and record its history"""
        self.bbox = bbox
        self.last_seen = frame_num
        self.appearances += 1
        self.skip_count = 0
        
        # Record center point for trajectory analysis
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.location_history.append((center_x, center_y, frame_num))
        
        # Limit history length
        if len(self.location_history) > 30:
            self.location_history.pop(0)
    
    def predict_next_bbox(self):
        """Predict next bounding box based on movement history"""
        if len(self.location_history) < 2:
            return self.bbox
        
        # Calculate velocity from last few positions
        history = self.location_history[-3:]
        if len(history) < 2:
            return self.bbox
        
        # Calculate average velocity
        deltas = []
        for i in range(1, len(history)):
            dx = history[i][0] - history[i-1][0]
            dy = history[i][1] - history[i-1][1]
            dt = history[i][2] - history[i-1][2]
            if dt > 0:
                deltas.append((dx/dt, dy/dt))
        
        if not deltas:
            return self.bbox
            
        avg_dx, avg_dy = np.mean(deltas, axis=0)
        
        # Predict new position based on velocity and time since last seen
        frames_since = self.skip_count
        x1, y1, x2, y2 = self.bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2 + avg_dx * frames_since
        center_y = (y1 + y2) / 2 + avg_dy * frames_since
        
        # Construct new bbox
        new_x1 = center_x - width / 2
        new_y1 = center_y - height / 2
        new_x2 = center_x + width / 2
        new_y2 = center_y + height / 2
        
        return [new_x1, new_y1, new_x2, new_y2]

class RobustFaceTracker:
    """Enhanced face tracker with robust matching strategies"""
    def __init__(self, max_frames_to_skip=MAX_FRAMES_TO_SKIP, 
                 iou_threshold=IOU_THRESHOLD, 
                 feature_match_threshold=FEATURE_MATCH_THRESHOLD,
                 starting_id=0):
        self.next_id = starting_id
        self.tracks = {}  # Active tracks
        self.max_frames_to_skip = max_frames_to_skip
        self.iou_threshold = iou_threshold
        self.feature_match_threshold = feature_match_threshold
        self.off_screen_tracks = {}
        self.frame_height = 0
        self.frame_width = 0
        
    def _calc_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        # Calculate coordinates of intersection
        xx1 = max(x1, x3)
        yy1 = max(y1, y3)
        xx2 = min(x2, x4)
        yy2 = min(y2, y4)
        # Calculate area of intersection and union
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        intersection = w * h
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union = box1_area + box2_area - intersection
        iou = intersection / max(union, 1e-6)
        return iou

    def _feature_similarity(self, feat1, feat2):
        """Calculate cosine similarity between two feature vectors"""
        similarity = 1 - cosine(feat1, feat2)
        return similarity

    def _is_near_border(self, bbox, margin=0.05):
        """Check if the bbox is near the edge of the frame"""
        x1, y1, x2, y2 = bbox
        margin_w = self.frame_width * margin
        margin_h = self.frame_height * margin
        
        # Check if the bbox is near any edge
        near_left = x1 < margin_w
        near_right = x2 > (self.frame_width - margin_w)
        near_top = y1 < margin_h
        near_bottom = y2 > (self.frame_height - margin_h)
        
        return near_left or near_right or near_top or near_bottom

    def _is_off_screen(self, bbox):
        """Check if the bbox is completely off screen"""
        x1, y1, x2, y2 = bbox
        return (x2 <= 0 or x1 >= self.frame_width or 
                y2 <= 0 or y1 >= self.frame_height)

    def update(self, faces, frame_num, frame_width, frame_height):
        """Update tracks with newly detected faces"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        face_to_track_map = {}
        
        # Handle case with no faces
        if not faces:
            for track_id, track in list(self.tracks.items()):
                track.skip_count += 1
                if track.skip_count > self.max_frames_to_skip:
                    # Check if the track went off screen
                    if self._is_near_border(track.bbox) or self._is_off_screen(track.bbox):
                        track.gone_off_screen = True
                        self.off_screen_tracks[track_id] = {
                            "entry_frame": track.first_seen,
                            "exit_frame": frame_num,
                            "appearances": track.appearances,
                            "best_quality_score": track.best_quality_score
                        }
                    del self.tracks[track_id]
            return face_to_track_map

        # If no existing tracks, create new tracks for all faces
        if not self.tracks:
            for i, face in enumerate(faces):
                track_id = self.next_id
                self.next_id += 1
                
                track = FaceTrack(
                    track_id=track_id,
                    bbox=face.bbox,
                    embedding=face.embedding if hasattr(face, 'embedding') else None,
                    frame_num=frame_num
                )
                self.tracks[track_id] = track
                face_to_track_map[i] = track_id
            return face_to_track_map

        # Multi-stage matching:
        # 1. Match by IoU (spatial continuity)
        # 2. Match by feature similarity (appearance)
        # 3. Create new tracks for unmatched faces
        
        # Prepare data for matching
        matched_faces = set()
        matched_tracks = set()
        
        # Stage 1: IoU matching
        for track_id, track in list(self.tracks.items()):
            if frame_num - track.last_seen > self.max_frames_to_skip:
                continue
                
            # Use predicted position for better matching
            predicted_bbox = track.predict_next_bbox()
            
            best_iou = self.iou_threshold
            best_match = -1
            
            for i, face in enumerate(faces):
                if i in matched_faces:
                    continue
                    
                iou = self._calc_iou(predicted_bbox, face.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            if best_match >= 0:
                matched_faces.add(best_match)
                matched_tracks.add(track_id)
                face_to_track_map[best_match] = track_id
                
                # Update track with new detection
                track.update_location(faces[best_match].bbox, frame_num)
                
                # Add new embedding if available
                if hasattr(faces[best_match], 'embedding') and faces[best_match].embedding is not None:
                    # Use detection score as a measure of quality
                    quality = getattr(faces[best_match], 'det_score', 0.5)
                    track.add_embedding(faces[best_match].embedding, quality)
        
        # Stage 2: Feature similarity matching for remaining unmatched faces
        if hasattr(faces[0], 'embedding'):  # Ensure embeddings are available
            for i, face in enumerate(faces):
                if i in matched_faces or not hasattr(face, 'embedding') or face.embedding is None:
                    continue
                    
                best_similarity = self.feature_match_threshold
                best_track = -1
                
                for track_id, track in self.tracks.items():
                    if (track_id in matched_tracks or 
                        not track.embeddings or
                        frame_num - track.last_seen > self.max_frames_to_skip):
                        continue
                    
                    # Compare with average embedding for robustness
                    avg_embedding = track.get_average_embedding()
                    if avg_embedding is None:
                        continue
                        
                    similarity = self._feature_similarity(face.embedding, avg_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_track = track_id
                
                if best_track >= 0:
                    matched_faces.add(i)
                    matched_tracks.add(best_track)
                    face_to_track_map[i] = best_track
                    
                    # Update track
                    self.tracks[best_track].update_location(face.bbox, frame_num)
                    quality = getattr(face, 'det_score', 0.5)
                    self.tracks[best_track].add_embedding(face.embedding, quality)
        
        # Stage 3: Create new tracks for unmatched faces
        for i, face in enumerate(faces):
            if i not in matched_faces:
                track_id = self.next_id
                self.next_id += 1
                
                track = FaceTrack(
                    track_id=track_id,
                    bbox=face.bbox,
                    embedding=face.embedding if hasattr(face, 'embedding') else None,
                    frame_num=frame_num
                )
                self.tracks[track_id] = track
                face_to_track_map[i] = track_id
        
        # Clean up lost tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id].skip_count += 1
                if self.tracks[track_id].skip_count > self.max_frames_to_skip:
                    # Check if track went off screen
                    if self._is_near_border(self.tracks[track_id].bbox) or self._is_off_screen(self.tracks[track_id].bbox):
                        self.tracks[track_id].gone_off_screen = True
                        self.off_screen_tracks[track_id] = {
                            "entry_frame": self.tracks[track_id].first_seen,
                            "exit_frame": frame_num,
                            "appearances": self.tracks[track_id].appearances,
                            "best_quality_score": self.tracks[track_id].best_quality_score
                        }
                    del self.tracks[track_id]
        
        return face_to_track_map
    
    def get_track_info(self, track_id):
        """Get information about a specific track"""
        return self.tracks.get(track_id, None)
    
    def get_all_active_tracks(self, current_frame):
        """Get all currently active tracks"""
        return {
            track_id: track for track_id, track in self.tracks.items()
            if current_frame - track.last_seen <= self.max_frames_to_skip
        }
    
    def get_off_screen_tracks(self):
        """Get tracks that went off screen"""
        return self.off_screen_tracks
    
    def save_state(self, filepath):
        """Save tracker state to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({'tracks': self.tracks, 'next_id': self.next_id,
                         'off_screen_tracks': self.off_screen_tracks}, f)
    
    def load_state(self, filepath):
        """Load tracker state from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.tracks = data['tracks']
                self.next_id = data['next_id']
                self.off_screen_tracks = data.get('off_screen_tracks', {})
            return True
        return False

class FaceProcessor:
    """Handles face extraction, alignment, and preprocessing"""
    def __init__(self, expansion_factor=EXPANSION_FACTOR):
        self.expansion_factor = expansion_factor
        
    def expand_bbox(self, bbox, img_width=0, img_height=0):
        """Expand bounding box while keeping it within image boundaries"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Expand more vertically to ensure entire face is captured
        new_width = width * self.expansion_factor
        new_height = height * self.expansion_factor * 1.1  # Extra vertical expansion
        
        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2
        
        # Ensure bbox stays within image boundaries
        if img_width > 0 and img_height > 0:
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            new_x2 = min(img_width, new_x2)
            new_y2 = min(img_height, new_y2)
        else:
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            
        return np.array([new_x1, new_y1, new_x2, new_y2])
    
    def align_face(self, img, landmarks, desired_size=(112, 112)):
        """Align face using 5-point landmarks"""
        if landmarks is None or len(landmarks) < 5:
            return None
        
        right_eye = landmarks[1]  # Right eye
        left_eye = landmarks[0]   # Left eye
        nose = landmarks[2]       # Nose
        mouth_right = landmarks[4]  # Right mouth corner
        mouth_left = landmarks[3]   # Left mouth corner
        
        # Calculate eye center and alignment parameters
        eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate scaling based on interpupillary distance
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = desired_size[0] * 0.33  # Target eye distance
        scale = desired_dist / max(dist, 1e-6)
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Adjust translation to center the face
        tX = desired_size[0] * 0.5
        tY = desired_size[1] * 0.4  # Position eyes slightly above the center
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply transformation
        aligned = cv2.warpAffine(img, M, desired_size, flags=cv2.INTER_CUBIC)
        return aligned
    
    def preprocess_face(self, face_img, desired_size=(112, 112)):
        """Apply preprocessing to improve recognition quality"""
        if face_img is None or face_img.size == 0:
            return None
            
        # Ensure correct size
        if face_img.shape[0] != desired_size[0] or face_img.shape[1] != desired_size[1]:
            face_img = cv2.resize(face_img, desired_size)
        
        # Apply histogram equalization to improve contrast
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            # Convert back to BGR
            face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return face_img
    
    def extract_face(self, img, face, desired_size=(112, 112)):
        """Extract and align face from image"""
        result = {
            "success": False,
            "method": None,
            "landmarks_detected": False,
            "aligned_face": None,
            "message": ""
        }
        
        try:
            img_height, img_width = img.shape[:2]
            
            # Method 1: Try alignment with landmarks if available
            if hasattr(face, 'landmark') and face.landmark is not None:
                landmarks = face.landmark
                result["landmarks_detected"] = True
                
                aligned_face = self.align_face(img, landmarks, desired_size)
                if aligned_face is not None:
                    result["success"] = True
                    result["method"] = "aligned_with_landmarks"
                    result["aligned_face"] = self.preprocess_face(aligned_face, desired_size)
                    result["message"] = "Successfully aligned face with landmarks"
                    return result
            
            # Method 2: Use expanded bounding box
            x1, y1, x2, y2 = map(float, face.bbox)
            expanded_bbox = self.expand_bbox(
                np.array([x1, y1, x2, y2]), 
                img_width=img_width, 
                img_height=img_height
            )
            
            x1, y1, x2, y2 = map(int, expanded_bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)
            
            if x2 <= x1 or y2 <= y1:
                result["message"] = "Invalid bounding box dimensions after expansion"
                return result
            
            face_img = img[y1:y2, x1:x2]
            if face_img.size == 0:
                result["message"] = "Bounding box crop resulted in empty image"
                return result
            
            # Apply preprocessing
            processed_face = self.preprocess_face(cv2.resize(face_img, desired_size), desired_size)
            
            if processed_face is not None:
                result["success"] = True
                result["method"] = "expanded_bbox"
                result["aligned_face"] = processed_face
                result["message"] = "Used expanded bounding box with preprocessing"
                return result
            else:
                result["message"] = "Failed to preprocess face image"
                return result
                
        except Exception as e:
            result["message"] = f"Error in face extraction: {str(e)}"
            return result

def draw_screen_border(frame, border_width=20, color=(0, 0, 255)):
    """Draw a red border at the edges of the frame.
    
    Args:
        frame: The video frame to draw on
        border_width: Width of the border in pixels
        color: Border color in BGR format (default is red: 0,0,255)
    
    Returns:
        The modified frame with border drawn
    """
    h, w = frame.shape[:2]
    
    # Draw horizontal borders (top and bottom)
    frame[:border_width, :] = color  # Top border
    frame[h-border_width:, :] = color  # Bottom border
    
    # Draw vertical borders (left and right)
    frame[:, :border_width] = color  # Left border
    frame[:, w-border_width:] = color  # Right border
    
    return frame

def check_border_crossing(track, frame_width, frame_height, border_width=20):
    """Check if a track's bounding box has crossed the screen border from outside to inside.

    Args:
        track: The face track object with location_history storing (x_center, y_center, frame_num)
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        border_width: Width of the border in pixels

    Returns:
        Boolean indicating if any part of the bounding box crossed into the screen from outside
    """
    if len(track.location_history) < 2:
        return False

    # Get current and previous bounding box from track
    current_bbox = track.bbox  # Assuming bbox is stored as [x1, y1, x2, y2]
    current_x1, current_y1, current_x2, current_y2 = current_bbox
    
    # Get previous position from history (assuming we need to reconstruct previous bbox)
    # Since location_history stores center points, we'll use the last known bbox size
    prev_center_x, prev_center_y, _ = track.location_history[-2]
    prev_width = current_x2 - current_x1  # Assuming width remains similar
    prev_height = current_y2 - current_y1  # Assuming height remains similar
    prev_x1 = prev_center_x - prev_width / 2
    prev_x2 = prev_center_x + prev_width / 2
    prev_y1 = prev_center_y - prev_height / 2
    prev_y2 = prev_center_y + prev_height / 2

    # Define border zones
    left_border = border_width
    right_border = frame_width - border_width
    top_border = border_width
    bottom_border = frame_height - border_width

    # Check if any part of the bbox crossed from outside to inside
    # Left border: was fully left of border, now any part is inside
    if prev_x2 <= left_border and current_x1 > left_border:
        return True
    
    # Right border: was fully right of border, now any part is inside
    if prev_x1 >= right_border and current_x2 < right_border:
        return True
    
    # Top border: was fully above border, now any part is inside
    if prev_y2 <= top_border and current_y1 > top_border:
        return True
    
    # Bottom border: was fully below border, now any part is inside
    if prev_y1 >= bottom_border and current_y2 < bottom_border:
        return True

    return False

class EnhancedFaceTrack(FaceTrack):
    """Extended FaceTrack class with border crossing detection"""
    def __init__(self, track_id, bbox, embedding=None, frame_num=0):
        super().__init__(track_id, bbox, embedding, frame_num)
        self.border_entries = []  # List of frame numbers where border was crossed
        
    def record_border_entry(self, frame_num):
        """Record a border entry event"""
        self.border_entries.append(frame_num)

def process_video(
        video_path: str,
        output_folder: str,
        stream_id: int = 0,
        desired_size: Tuple[int, int] = (112, 112),
        expansion_factor: float = EXPANSION_FACTOR,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
        skip_frames: int = 1,
        save_video: bool = True,  # Changed default to True to save automatically
        max_frames: int = None,
        max_frames_to_skip: int = MAX_FRAMES_TO_SKIP,
        resume_processing: bool = False,
        border_width: int = 20  # Added parameter for border width
) -> None:
    # Setup output directories
    stream_folder = os.path.join(output_folder, f"stream_{stream_id}")
    Path(stream_folder).mkdir(parents=True, exist_ok=True)
    faces_output_folder = os.path.join(stream_folder, "faces")
    Path(faces_output_folder).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(stream_folder, "processing_log.txt")
    entry_times_path = os.path.join(stream_folder, "entry_times.txt")
    tracker_state_path = os.path.join(stream_folder, "tracker_state.pkl")
    
    logger.info(f"Processing video: {video_path} (Stream ID: {stream_id})")
    logger.info(f"Output directory: {stream_folder}")
    
    # Initialize components
    face_analyzer = FaceAnalyzer(det_size=(640, 640))
    face_processor = FaceProcessor(expansion_factor=expansion_factor)
    
    # Find the next available track ID
    starting_id = 0
    if os.path.exists(faces_output_folder):
        existing_tracks = [
            int(d.split('_')[1]) for d in os.listdir(faces_output_folder)
            if d.startswith('track_') and os.path.isdir(os.path.join(faces_output_folder, d))
        ]
        starting_id = max(existing_tracks, default=-1) + 1
    
    # Override the RobustFaceTracker class to use EnhancedFaceTrack
    class EnhancedTracker(RobustFaceTracker):
        def update(self, faces, frame_num, frame_width, frame_height):
            # Create EnhancedFaceTrack objects instead of regular FaceTrack
            original_init = FaceTrack.__init__
            def enhanced_init(self, track_id, bbox, embedding=None, frame_num=0):
                self.border_entries = []
                original_init(self, track_id, bbox, embedding, frame_num)
            
            # Save original init
            temp = FaceTrack.__init__
            # Override init temporarily
            FaceTrack.__init__ = enhanced_init
            
            # Call parent's update
            result = super().update(faces, frame_num, frame_width, frame_height)
            
            # Restore original init
            FaceTrack.__init__ = temp
            
            # Check for border crossings in active tracks
            for track_id, track in self.tracks.items():
                if check_border_crossing(track, frame_width, frame_height, border_width):
                    if not hasattr(track, 'border_entries'):
                        track.border_entries = []
                    track.border_entries.append(frame_num)
                    logger.info(f"Track {track_id} entered through border at frame {frame_num}")
            
            return result
    
    # Initialize tracker with enhanced class
    tracker = EnhancedTracker(
        max_frames_to_skip=max_frames_to_skip,
        iou_threshold=IOU_THRESHOLD,
        feature_match_threshold=FEATURE_MATCH_THRESHOLD,
        starting_id=starting_id
    )
    
    # Try to load previous state if resuming
    last_processed_frame = 0
    if resume_processing and os.path.exists(tracker_state_path):
        if tracker.load_state(tracker_state_path):
            logger.info(f"Resumed tracking from saved state. Next track ID: {tracker.next_id}")
            
            # Find the last processed frame number
            with open(log_path, 'r') as log:
                for line in log:
                    if "Frame " in line:
                        try:
                            frame_num = int(line.split("Frame ")[1].split(":")[0].split(",")[0])
                            last_processed_frame = max(last_processed_frame, frame_num)
                        except:
                            pass
            logger.info(f"Resuming from frame {last_processed_frame + 1}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize video writer - always save the video
    timestamp = int(time.time())
    output_video_path = os.path.join(stream_folder, f"processed_video_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize counters
    track_image_counts = defaultdict(int)
    for track_dir in os.listdir(faces_output_folder):
        if track_dir.startswith('track_'):
            track_id = int(track_dir.split('_')[1])
            image_count = len([f for f in os.listdir(os.path.join(faces_output_folder, track_dir)) 
                              if f.endswith(('.jpg', '.png'))])
            track_image_counts[track_id] = image_count
    
    # Skip to last processed frame if resuming
    if last_processed_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_processed_frame)
    
    # Main processing loop
    frame_idx = last_processed_frame
    processed_frames = 0
    pbar = tqdm(total=total_frames - last_processed_frame if max_frames is None 
                else min(total_frames - last_processed_frame, max_frames))
    
    with open(log_path, 'a') as log:
        log.write(f"\nProcessing video: {video_path} (Stream ID: {stream_id})\n")
        log.write(f"Parameters: expansion_factor={expansion_factor}, confidence_threshold={confidence_threshold}, "
                 f"high_confidence_threshold={high_confidence_threshold}, starting_id={starting_id}\n")
        log.write(f"Starting from frame: {frame_idx}\n")
        log.write("=" * 80 + "\n")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            pbar.update(1)
            frame_idx += 1
            
            if max_frames is not None and processed_frames >= max_frames:
                break
            
            if (frame_idx - last_processed_frame) % skip_frames == 0:
                processed_frames += 1
                
                # Draw border on frame
                frame_with_border = draw_screen_border(frame.copy(), border_width=(border_width))
                
                # Convert to RGB for face detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                faces = face_analyzer.get_faces(frame_rgb, min_confidence=confidence_threshold)
                
                # Update tracks
                face_track_map = tracker.update(faces, frame_idx, width, height)
                
                if not faces:
                    with open(log_path, 'a') as log:
                        log.write(f"Frame {frame_idx}: No faces detected above tracking threshold ({confidence_threshold})\n")
                
                # Process high-confidence faces for saving
                for face_idx, track_id in face_track_map.items():
                    face = faces[face_idx]
                    track = tracker.get_track_info(track_id)
                    
                    # Check for border entry events
                    if hasattr(track, 'border_entries') and track.border_entries and frame_idx in track.border_entries:
                        with open(log_path, 'a') as log:
                            log.write(f"Frame {frame_idx}, Track {track_id}: ENTERED through border\n")
                    
                    # Only process high confidence detections for saving
                    if hasattr(face, 'det_score') and face.det_score >= high_confidence_threshold:
                        # Extract and align face
                        result = face_processor.extract_face(frame, face, desired_size)
                        
                        if result["success"]:
                            # Create directory for this track if it doesn't exist
                            track_dir = os.path.join(faces_output_folder, f"track_{track_id}")
                            os.makedirs(track_dir, exist_ok=True)
                            
                            # Generate unique filename
                            image_count = track_image_counts[track_id]
                            image_path = os.path.join(track_dir, f"frame_{frame_idx:06d}_{image_count:03d}.jpg")
                            
                            # Save the face image
                            cv2.imwrite(image_path, result["aligned_face"])
                            track_image_counts[track_id] += 1
                            
                            # Update best quality face if needed
                            if track and face.det_score > track.best_quality_score:
                                track.best_quality_score = face.det_score
                                track.best_face_image = result["aligned_face"]
                                track.best_face_frame = frame_idx
                            
                            # Check if track has entered through border
                            entered_through_border = hasattr(track, 'border_entries') and len(track.border_entries) > 0
                            
                            with open(log_path, 'a') as log:
                                log.write(f"Frame {frame_idx}, Track {track_id}:\n")
                                log.write(f"  Confidence: {face.det_score:.3f}\n")
                                log.write(f"  Method: {result['method']}\n")
                                log.write(f"  Landmarks detected: {result['landmarks_detected']}\n")
                                log.write(f"  Came in through border: {entered_through_border}\n")  # New entry
                                log.write(f"  Saved to: {image_path}\n")
                                log.write("-" * 80 + "\n")
                        else:
                            entered_through_border = hasattr(track, 'border_entries') and len(track.border_entries) > 0
                            with open(log_path, 'a') as log:
                                log.write(f"Frame {frame_idx}, Track {track_id}: Failed to extract - {result['message']}\n")
                                log.write(f"  Came in through border: {entered_through_border}\n")  # New entry
                                log.write("-" * 80 + "\n")
                    else:
                        # Log detection below high confidence threshold
                        entered_through_border = hasattr(track, 'border_entries') and len(track.border_entries) > 0
                        with open(log_path, 'a') as log:
                            log.write(f"Frame {frame_idx}, Track {track_id}: Detected (confidence: {face.det_score:.3f}) "
                                     f"but not saved (below {high_confidence_threshold})\n")
                            log.write(f"  Came in through border: {entered_through_border}\n")  # New entry
                
                # Draw bounding boxes and track info on frame
                # We always save the video now
                
                # Draw all active tracks
                active_tracks = tracker.get_all_active_tracks(frame_idx)
                for track_id, track in active_tracks.items():
                    if frame_idx - track.last_seen <= max_frames_to_skip:
                        x1, y1, x2, y2 = map(int, track.bbox)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        # Generate consistent color based on track ID
                        color = ((track_id * 137) % 255, (track_id * 97) % 255, (track_id * 227) % 255)
                        
                        # Draw rectangle and ID
                        if x2 > x1 and y2 > y1:
                            cv2.rectangle(frame_with_border, (x1, y1), (x2, y2), color, 2)
                            
                            # Add text with track ID, confidence, and entry status
                            entered = hasattr(track, 'border_entries') and len(track.border_entries) > 0
                            label = f"ID: {track_id}"
                            if entered:
                                label += " [ENTERED]"
                                
                            face_idx = [i for i, t_id in face_track_map.items() if t_id == track_id]
                            if face_idx:
                                face = faces[face_idx[0]]
                                if hasattr(face, 'det_score'):
                                    label += f" ({face.det_score:.2f})"
                            
                            cv2.putText(frame_with_border, label, (x1, y1 - 10 if y1 - 10 >= 0 else y1 + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Draw track history (trajectory)
                            if len(track.location_history) > 1:
                                points = [(int(x), int(y)) for x, y, _ in track.location_history]
                                for i in range(1, len(points)):
                                    cv2.line(frame_with_border, points[i-1], points[i], color, 1)
                
                # Add frame number and active track count
                cv2.putText(frame_with_border, f"Frame: {frame_idx} | Tracks: {len(active_tracks)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame to output video
                output_video.write(frame_with_border)
                
                # Save tracker state periodically
                if processed_frames % 100 == 0:
                    tracker.save_state(tracker_state_path)
            
        # End of video processing
        
        # Save final tracker state
        tracker.save_state(tracker_state_path)
        
        # Write entry/exit log with new border entry information
        off_screen_tracks = tracker.get_off_screen_tracks()
        if off_screen_tracks:
            with open(entry_times_path, 'a') as et:
                et.write(f"\nVideo: {video_path} (Stream ID: {stream_id})\n")
                et.write("Track ID | Entry Frame | Exit Frame | Appearances | Quality | Entered Through Border\n")
                et.write("-" * 80 + "\n")
                for track_id, info in off_screen_tracks.items():
                    # Check if this track has border entry events
                    track = tracker.get_track_info(track_id)
                    entered_through_border = "Yes" if (track and hasattr(track, 'border_entries') and 
                                                      len(track.border_entries) > 0) else "No"
                    
                    et.write(f"{track_id} | {info['entry_frame']} | {info['exit_frame']} | "
                            f"{info['appearances']} | {info['best_quality_score']:.3f} | {entered_through_border}\n")
        
        # Save best quality face for each track
        for track_id, track in tracker.tracks.items():
            if track.best_face_image is not None:
                track_dir = os.path.join(faces_output_folder, f"track_{track_id}")
                os.makedirs(track_dir, exist_ok=True)
                best_face_path = os.path.join(track_dir, f"best_quality.jpg")
                cv2.imwrite(best_face_path, track.best_face_image)
        
        # Write processing summary
        with open(log_path, 'a') as log:
            log.write("\nProcessing Summary:\n")
            log.write(f"Total frames processed: {processed_frames} / {total_frames}\n")
            log.write(f"New tracks added: {tracker.next_id - starting_id}\n")
            log.write(f"Total tracks: {tracker.next_id}\n")
            log.write(f"Tracks exited off-screen: {len(off_screen_tracks)}\n")
            log.write(f"Tracking threshold: {confidence_threshold}, Saving threshold: {high_confidence_threshold}\n")
            log.write("Track image counts (high confidence only):\n")
            for track_id, count in sorted(track_image_counts.items()):
                track = tracker.get_track_info(track_id)
                entered = "Yes" if (track and hasattr(track, 'border_entries') and 
                                  len(track.border_entries) > 0) else "No"
                log.write(f"  Track {track_id}: {count} images, Entered through border: {entered}\n")
    
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        # Save tracker state on error to allow resuming
        tracker.save_state(tracker_state_path)
    
    finally:
        # Clean up resources
        cap.release()
        if output_video is not None:
            output_video.release()
        pbar.close()
    
    logger.info(f"Video processing completed. Results saved to {stream_folder}")
    logger.info(f"Tracked {tracker.next_id - starting_id} faces. Total images saved: {sum(track_image_counts.values())}")
    logger.info(f"Tracks exited off-screen: {len(off_screen_tracks)}. See {entry_times_path} for details.")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description="Enhanced video face detection, tracking, and alignment pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output directory for processed images and video")
    parser.add_argument("--stream-id", "-s", type=int, default=0, help="Stream ID to organize output (default: 0)")
    parser.add_argument("--size", type=int, default=112, help="Output face image size (width and height)")
    parser.add_argument("--expansion", "-e", type=float, default=EXPANSION_FACTOR, help="Factor by which to expand face crop boundary")
    parser.add_argument("--confidence", "-c", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Minimum confidence threshold for tracking")
    parser.add_argument("--high-confidence", "-hc", type=float, default=HIGH_CONFIDENCE_THRESHOLD, help="Minimum confidence threshold for saving faces")
    parser.add_argument("--skip-frames", "-sf", type=int, default=1, help="Process every nth frame (1 = process all frames)")
    parser.add_argument("--no-save-video", action="store_true", help="Disable saving processed video with bounding boxes")
    parser.add_argument("--max-frames", "-mf", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--max-skip", "-ms", type=int, default=MAX_FRAMES_TO_SKIP, help="Maximum frames to skip for tracking")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from previous processing state if available")
    parser.add_argument("--border-width", "-bw", type=int, default=20, help="Width of the border for entry/exit detection")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Process the video
    process_video(
        video_path=args.input,
        output_folder=args.output,
        stream_id=args.stream_id,
        desired_size=(args.size, args.size),
        expansion_factor=args.expansion,
        confidence_threshold=args.confidence,
        high_confidence_threshold=args.high_confidence,
        skip_frames=args.skip_frames,
        save_video=not args.no_save_video,
        max_frames=args.max_frames,
        max_frames_to_skip=args.max_skip,
        resume_processing=args.resume,
        border_width=args.border_width
    )

if __name__ == "__main__":
    main()