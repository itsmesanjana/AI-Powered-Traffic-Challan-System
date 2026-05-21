from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Download the pre-trained YOLOv8 model for license plate detection
# You can use a custom trained model or a general object detection model
# For this example, we'll use a general object detection model and fine-tune it for license plates
# or use a pre-trained model specifically for license plate detection

# Model path - you can download a pre-trained model from the Ultralytics hub
# or use a custom trained model
# Use a license plate specific model for better accuracy
MODEL_PATH = "best.pt"  # You can download a license plate detection model
# If the specific model is not available, fall back to general model
FALLBACK_MODEL = "yolov8n.pt"

class YOLOPlateDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the YOLO plate detector.
        
        Args:
            model_path (str): Path to the YOLO model file
        """
        try:
            self.model = YOLO(model_path)
        except Exception:
            # Fall back to general model if specific plate model is not available
            print(f"Warning: Could not load {model_path}, falling back to general model")
            self.model = YOLO(FALLBACK_MODEL)
        self.confidence_threshold = 0.7  # Higher confidence for plate detection
        
    def detect_number_plate(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Detect probable number plate region using YOLO.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            (x, y, w, h) tuple for the detected plate region, or None if not found
        """
        # Run YOLO inference
        results = self.model(image, conf=self.confidence_threshold)
        
        if results and len(results[0].boxes) > 0:
            # Get the box with highest confidence
            best_box = results[0].boxes[0]
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            
            # Convert to (x, y, w, h) format
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            # Crop the plate region for better OCR
            cropped_plate = image[y:y+h, x:x+w]
            
            # Apply preprocessing for better OCR results
            gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
            # Apply some preprocessing to enhance the plate region
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (should be the plate)
                largest_contour = max(contours, key=cv2.contourArea)
                x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(largest_contour)
                
                # Adjust coordinates relative to original image
                final_x = x + x_plate
                final_y = y + y_plate
                final_w = w_plate
                final_h = h_plate
                
                return (final_x, final_y, final_w, final_h)
            
            return (x, y, w, h)
        
        return None
    
    def draw_plate_box(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
        """
        Draw bounding box around detected plate.
        
        Args:
            image: Input image
            bbox: Bounding box coordinates (x, y, w, h) or None
            
        Returns:
            Image with bounding box drawn
        """
        output = image.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return output

# Create a global instance of the YOLO detector
yolo_detector = YOLOPlateDetector()

def detect_number_plate(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Detect probable number plate region using YOLO.
    Returns: (x, y, w, h) or None
    """
    return yolo_detector.detect_number_plate(image)

def draw_plate_box(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """
    Draw bounding box around detected plate.
    """
    return yolo_detector.draw_plate_box(image, bbox)
