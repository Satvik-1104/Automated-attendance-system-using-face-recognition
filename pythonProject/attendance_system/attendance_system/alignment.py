import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis  # Use InsightFace's RetinaFace
from PIL import Image
from pillow_heif import register_heif_opener
from typing import Tuple, Optional, Dict, Any

register_heif_opener()

# Initialize RetinaFace
app = FaceAnalysis(name='buffalo_l')  # Lightweight RetinaFace model
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for CPU, -1 for GPU
EXPANSION_FACTOR = 1.2


def load_heic_image(heic_file: str) -> np.ndarray:
    """Load HEIC image and convert to BGR format."""
    image = Image.open(heic_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def expand_bbox(
        bbox: np.ndarray,
        expansion_factor: float = EXPANSION_FACTOR,
        img_width: int = 0,
        img_height: int = 0
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Calculate center of the bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate new size
    new_width = width * expansion_factor
    new_height = height * expansion_factor

    # Calculate new bbox coordinates
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # Ensure bbox stays within image boundaries if dimensions are provided
    if img_width > 0 and img_height > 0:
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_width, new_x2)
        new_y2 = min(img_height, new_y2)

    return np.array([new_x1, new_y1, new_x2, new_y2])


def align_face(
        img: np.ndarray,
        right_eye: Tuple[float, float],
        left_eye: Tuple[float, float],
        nose: Tuple[float, float],
        mouth_right: Tuple[float, float],
        mouth_left: Tuple[float, float],
        desired_size: Tuple[int, int] = (112, 112),
        expansion_factor: float = EXPANSION_FACTOR
) -> Optional[np.ndarray]:
    try:
        # Calculate center point between eyes
        eyes_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2
        )

        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate scale based on distance between eyes but with expansion factor
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = desired_size[0] * 0.3  # Reduced from 0.4 to allow more context
        scale = desired_dist / max(dist, 1e-6) / expansion_factor  # Apply expansion factor

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update translation to center the face
        tX = desired_size[0] * 0.5
        tY = desired_size[1] * 0.33  # Position eyes slightly above center
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        # Apply affine transformation
        aligned = cv2.warpAffine(img, M, desired_size, flags=cv2.INTER_CUBIC)
        return aligned
    except Exception as e:
        print(f"Alignment failed: {str(e)}")
        return None


def process_image(
        img_path: str,
        output_path: str,
        desired_size: Tuple[int, int] = (112, 112),
        expansion_factor: float = EXPANSION_FACTOR,
        confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    result = {
        "success": False,
        "method": None,
        "confidence": None,
        "face_count": 0,
        "landmarks_detected": False,
        "message": ""
    }

    try:
        # Load image (support HEIC)
        if img_path.lower().endswith('.heic'):
            img = load_heic_image(img_path)
        else:
            img = cv2.imread(img_path)

        if img is None:
            result["message"] = f"Could not load image: {img_path}"
            return result

        # Get image dimensions
        img_height, img_width = img.shape[:2]

        # Try with different image dimensions for detection
        img_sizes = [(640, 640), (320, 320), (1280, 1280)]
        faces = []
        detection_confidence = 0

        # Convert to RGB for insightface
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Try different detection sizes
        for det_size in img_sizes:
            # Set detection size
            app.prepare(ctx_id=0, det_size=det_size)
            # Detect faces
            faces = app.get(img_rgb)

            # If faces detected, break the loop
            if faces and len(faces) > 0:
                # Get detection confidence from the first face
                if hasattr(faces[0], 'det_score'):
                    detection_confidence = faces[0].det_score
                break

        # Store face count and confidence
        result["face_count"] = len(faces) if faces else 0
        result["confidence"] = detection_confidence

        # Still no faces detected after trying different sizes or confidence too low
        if not faces or len(faces) == 0 or detection_confidence < confidence_threshold:
            # As a fallback, just save a resized version of the original image
            resized_img = cv2.resize(img, desired_size)
            cv2.imwrite(output_path, resized_img)

            result[
                "message"] = f"No faces with sufficient confidence detected in {img_path}, saved resized original image instead"
            result["method"] = "resized_original"
            result["success"] = True
            return result

        # Find largest face based on bounding box area
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Check if landmarks exist
        if not hasattr(largest_face, 'landmark') or largest_face.landmark is None:
            # Extract face using expanded bounding box instead
            x1, y1, x2, y2 = map(float, largest_face.bbox)

            # Expand the bounding box by the expansion factor
            expanded_bbox = expand_bbox(
                np.array([x1, y1, x2, y2]),
                expansion_factor=expansion_factor,
                img_width=img_width,
                img_height=img_height
            )

            x1, y1, x2, y2 = map(int, expanded_bbox)

            # Ensure coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            face_img = img[y1:y2, x1:x2]

            # Resize to desired size
            if face_img.size > 0:  # Check if crop is valid
                face_img = cv2.resize(face_img, desired_size)
                cv2.imwrite(output_path, face_img)

                result["success"] = True
                result["method"] = "expanded_bbox_no_landmarks"
                result["message"] = f"No landmarks detected in {img_path}, used expanded bounding box"
                return result
            else:
                # If crop is invalid, save resized original
                resized_img = cv2.resize(img, desired_size)
                cv2.imwrite(output_path, resized_img)

                result["success"] = True
                result["method"] = "resized_original_after_invalid_bbox"
                result["message"] = f"Invalid bounding box in {img_path}, saved resized original"
                return result

        # Extract facial landmarks
        landmarks = largest_face.landmark  # 5-point landmarks
        right_eye = landmarks[1]  # Right eye
        left_eye = landmarks[0]  # Left eye
        nose = landmarks[2]  # Nose
        mouth_right = landmarks[4]  # Right mouth corner
        mouth_left = landmarks[3]  # Left mouth corner

        result["landmarks_detected"] = True

        # Align face with expanded boundaries
        aligned_face = align_face(
            img,
            right_eye,
            left_eye,
            nose,
            mouth_right,
            mouth_left,
            desired_size,
            expansion_factor
        )

        if aligned_face is not None:
            cv2.imwrite(output_path, aligned_face)

            result["success"] = True
            result["method"] = "aligned_with_landmarks"
            result["message"] = f"Successfully aligned face with landmarks in {img_path}"
            return result
        else:
            # Fallback: extract face using expanded bounding box
            x1, y1, x2, y2 = map(float, largest_face.bbox)

            # Expand the bounding box by the expansion factor
            expanded_bbox = expand_bbox(
                np.array([x1, y1, x2, y2]),
                expansion_factor=expansion_factor,
                img_width=img_width,
                img_height=img_height
            )

            x1, y1, x2, y2 = map(int, expanded_bbox)

            # Ensure coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            face_img = img[y1:y2, x1:x2]

            # Resize to desired size
            if face_img.size > 0:  # Check if crop is valid
                face_img = cv2.resize(face_img, desired_size)
                cv2.imwrite(output_path, face_img)

                result["success"] = True
                result["method"] = "expanded_bbox_after_alignment_failure"
                result["message"] = f"Alignment failed for {img_path}, used expanded bounding box"
                return result
            else:
                # Last resort: save resized original
                resized_img = cv2.resize(img, desired_size)
                cv2.imwrite(output_path, resized_img)

                result["success"] = True
                result["method"] = "resized_original_last_resort"
                result["message"] = f"All processing attempts failed for {img_path}, saved resized original"
                return result

    except Exception as e:
        # Attempt to save a resized version of the original image as fallback
        try:
            if 'img' in locals() and img is not None:
                resized_img = cv2.resize(img, desired_size)
                cv2.imwrite(output_path, resized_img)

                result["success"] = True
                result["method"] = "resized_original_after_error"
                result["message"] = f"Error: {str(e)}. Saved resized original for {img_path}"
                return result
        except Exception as fallback_error:
            result[
                "message"] = f"All processing failed for {img_path}: {str(e)}, fallback also failed: {str(fallback_error)}"
            return result

    return result


def process_dataset(
        input_root: str,
        output_root: str,
        desired_size: Tuple[int, int] = (112, 112),
        expansion_factor: float = EXPANSION_FACTOR,
        confidence_threshold: float = 0.5,
        log_file: str = "processing_log.txt"
) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_root, exist_ok=True)

    # Create log file
    log_path = os.path.join(output_root, log_file)

    with open(log_path, 'w') as log:
        log.write(
            f"Processing started with expansion_factor={expansion_factor}, confidence_threshold={confidence_threshold}\n")
        log.write("=" * 80 + "\n")

    student_folders = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

    for student in tqdm(student_folders, desc="Processing students"):
        input_dir = os.path.join(input_root, student)
        output_dir = os.path.join(output_root, student)

        os.makedirs(output_dir, exist_ok=True)

        image_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.heic'))
        ]

        for img_file in tqdm(image_files, desc=f"Processing {student}", leave=False):
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file.rsplit('.', 1)[0] + '.jpg')  # Convert to JPG

            if not os.path.exists(output_path):
                result = process_image(
                    input_path,
                    output_path,
                    desired_size,
                    expansion_factor,
                    confidence_threshold
                )

                # Log the results
                with open(log_path, 'a') as log:
                    log.write(f"File: {input_path}\n")
                    log.write(f"  Success: {result['success']}\n")
                    log.write(f"  Method: {result['method']}\n")
                    log.write(f"  Face count: {result['face_count']}\n")
                    log.write(f"  Detection confidence: {result['confidence']}\n")
                    log.write(f"  Landmarks detected: {result['landmarks_detected']}\n")
                    log.write(f"  Message: {result['message']}\n")
                    log.write("-" * 80 + "\n")

                if not result["success"]:
                    print(f"Failed to process {input_path}: {result['message']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection and alignment pipeline")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input directory containing student folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory for processed images"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=112,
        help="Output image size (width and height)"
    )
    parser.add_argument(
        "--expansion",
        type=float,
        default=1.2,
        help="Factor by which to expand face crop boundary (e.g., 1.5 = 50% larger)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for face detection"
    )

    args = parser.parse_args()

    process_dataset(
        input_root=args.input,
        output_root=args.output,
        desired_size=(args.size, args.size),
        expansion_factor=args.expansion,
        confidence_threshold=args.confidence
    )
