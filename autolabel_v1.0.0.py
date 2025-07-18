import streamlit as st
import cv2
import os
import json
import numpy as np
from pathlib import Path
import tempfile
import zipfile
from datetime import datetime
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io


try:
    from ultralytics import YOLO
except ImportError:
    st.error("Please install ultralytics: pip install ultralytics")
    st.stop()

# Configuration
CONSTRUCTION_CLASSES = {
    'person': 0,
    'hard_hat': 1, 
    'safety_vest': 2,
    'excavator': 3,
    'truck': 4,
    'crane': 5,
    'concrete_mixer': 6,
    'scaffolding': 7
}

CONFIDENCE_THRESHOLDS = {
    'auto_approve': 0.85,
    'review_required': 0.6,
    'reject': 0.3
}

class ConstructionAutoLabeler:
    def __init__(self):
        self.model = None
        self.results_cache = {}
        
    @st.cache_resource
    def load_model(_self, model_size='n'):
        """Load YOLOv12 model"""
        try:
            # Try YOLOv12 first, fallback to YOLOv8 if not available
            model_name = f'yolo12{model_size}.pt'
            model = YOLO(model_name)
            st.success(f"Loaded YOLOv12{model_size} model")
            return model
        except:
            try:
                model_name = f'yolov8{model_size}.pt'
                model = YOLO(model_name)
                st.warning(f"YOLOv12 not available, using YOLOv8{model_size}")
                return model
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return None
    
    def extract_frames(self, video_path, fps=2):
        """Extract frames from video at specified fps"""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        frames = []
        frame_count = 0
        success = True
        
        while success:
            success, frame = cap.read()
            if success and frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / video_fps,
                    'image': frame_rgb
                })
            frame_count += 1
            
        cap.release()
        return frames
    
    def predict_frame(self, image, confidence_threshold=0.5):
        """Run YOLO prediction on a single frame"""
        if self.model is None:
            return None
            
        results = self.model(image, conf=confidence_threshold)
        return results[0] if results else None
    
    def categorize_predictions(self, predictions):
        """Categorize predictions based on confidence thresholds"""
        if predictions is None or predictions.boxes is None:
            return {'auto_approve': [], 'review': [], 'reject': []}
        
        categorized = {'auto_approve': [], 'review': [], 'reject': []}
        
        boxes = predictions.boxes
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            box_data = {
                'bbox': boxes.xyxy[i].tolist(),
                'confidence': conf,
                'class_id': int(boxes.cls[i]),
                'class_name': predictions.names[int(boxes.cls[i])]
            }
            
            if conf >= CONFIDENCE_THRESHOLDS['auto_approve']:
                categorized['auto_approve'].append(box_data)
            elif conf >= CONFIDENCE_THRESHOLDS['review_required']:
                categorized['review'].append(box_data)
            else:
                categorized['reject'].append(box_data)
                
        return categorized
    
    def draw_predictions(self, image, predictions, category='all'):
        """Draw bounding boxes on image"""
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        colors = {
            'auto_approve': 'green',
            'review': 'orange', 
            'reject': 'red'
        }
        
        for cat, preds in predictions.items():
            if category != 'all' and category != cat:
                continue
                
            color = colors.get(cat, 'blue')
            for pred in preds:
                bbox = pred['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label = f"{pred['class_name']}: {pred['confidence']:.2f}"
                draw.text((x1, y1-20), label, fill=color)
        
        return np.array(img_pil)
    
    def export_annotations(self, all_predictions, format_type='coco'):
        """Export annotations in various formats"""
        if format_type == 'coco':
            return self._export_coco(all_predictions)
        elif format_type == 'yolo':
            return self._export_yolo(all_predictions)
        elif format_type == 'csv':
            return self._export_csv(all_predictions)
    
    def _export_coco(self, all_predictions):
        """Export in COCO format"""
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for name, i in CONSTRUCTION_CLASSES.items()]
        }
        
        annotation_id = 1
        for frame_idx, (frame_info, predictions) in enumerate(all_predictions.items()):
            # Add image info
            coco_data["images"].append({
                "id": frame_idx,
                "file_name": f"frame_{frame_idx}.jpg",
                "width": 640,  # You might want to get actual dimensions
                "height": 480
            })
            
            # Add annotations
            for pred in predictions.get('auto_approve', []):
                bbox = pred['bbox']
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": frame_idx,
                    "category_id": pred['class_id'],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1
                
        return json.dumps(coco_data, indent=2)
    
    def _export_csv(self, all_predictions):
        """Export as CSV"""
        rows = []
        for frame_idx, (frame_info, predictions) in enumerate(all_predictions.items()):
            for pred in predictions.get('auto_approve', []):
                bbox = pred['bbox']
                rows.append({
                    'frame_number': frame_idx,
                    'class_name': pred['class_name'],
                    'confidence': pred['confidence'],
                    'x1': bbox[0],
                    'y1': bbox[1], 
                    'x2': bbox[2],
                    'y2': bbox[3]
                })
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

def main():
    st.set_page_config(
        page_title="Construction Video Autolabeling",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Construction Video Autolabeling Pipeline")
    st.markdown("Upload construction site videos and get automated object detection labels!")
    
    # Initialize the autolabeler
    labeler = ConstructionAutoLabeler()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    model_size = st.sidebar.selectbox("Model Size", ['n', 's', 'm', 'l', 'x'], index=0)
    fps = st.sidebar.slider("Frame Extraction FPS", 1, 5, 2)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5)
    
    # Load model
    if labeler.model is None:
        with st.spinner("Loading YOLO model..."):
            labeler.model = labeler.load_model(model_size)
    
    if labeler.model is None:
        st.error("Failed to load model. Please check your installation.")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload construction site video for autolabeling"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.success(f"Uploaded: {uploaded_file.name}")
            
            # Process video button
            if st.button("🚀 Process Video", type="primary"):
                with st.spinner("Extracting frames..."):
                    frames = labeler.extract_frames(video_path, fps)
                    st.session_state.frames = frames
                    st.session_state.video_name = uploaded_file.name
                
                with st.spinner("Running autolabeling..."):
                    all_predictions = {}
                    progress_bar = st.progress(0)
                    
                    for i, frame_data in enumerate(frames):
                        image = frame_data['image']
                        predictions = labeler.predict_frame(image, confidence_threshold)
                        categorized = labeler.categorize_predictions(predictions)
                        
                        all_predictions[f"frame_{i}"] = categorized
                        progress_bar.progress((i + 1) / len(frames))
                    
                    st.session_state.all_predictions = all_predictions
                    st.session_state.processed = True
                
                st.success(f"Processed {len(frames)} frames!")
    
    with col2:
        st.header("Results")
        
        if 'processed' in st.session_state and st.session_state.processed:
            frames = st.session_state.frames
            all_predictions = st.session_state.all_predictions
            
            # Summary statistics
            total_auto = sum(len(pred['auto_approve']) for pred in all_predictions.values())
            total_review = sum(len(pred['review']) for pred in all_predictions.values())
            total_reject = sum(len(pred['reject']) for pred in all_predictions.values())
            
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Auto-Approved", total_auto, delta="✅")
            with col2b:
                st.metric("Needs Review", total_review, delta="⚠️")
            with col2c:
                st.metric("Low Confidence", total_reject, delta="❌")
            
            # Frame selector
            frame_idx = st.slider("Select Frame", 0, len(frames)-1, 0)
            
            # Display options
            display_category = st.selectbox(
                "Show Predictions", 
                ['all', 'auto_approve', 'review', 'reject']
            )
            
            # Display frame with predictions
            if frame_idx < len(frames):
                frame_data = frames[frame_idx]
                predictions = all_predictions[f"frame_{frame_idx}"]
                
                # Draw predictions on image
                annotated_image = labeler.draw_predictions(
                    frame_data['image'], 
                    predictions, 
                    display_category
                )
                
                st.image(annotated_image, caption=f"Frame {frame_idx} - Timestamp: {frame_data['timestamp']:.2f}s")
                
                # Show prediction details
                if predictions:
                    st.subheader("Detection Details")
                    for category, preds in predictions.items():
                        if preds and (display_category == 'all' or display_category == category):
                            st.write(f"**{category.replace('_', ' ').title()}** ({len(preds)} detections)")
                            for pred in preds:
                                st.write(f"- {pred['class_name']}: {pred['confidence']:.3f}")
            
            # Export section
            st.header("Export Results")
            export_format = st.selectbox("Export Format", ['coco', 'csv', 'yolo'])
            
            if st.button("📥 Generate Export"):
                with st.spinner("Generating export..."):
                    export_data = labeler.export_annotations(all_predictions, export_format)
                    
                    if export_format == 'csv':
                        st.download_button(
                            label="Download CSV",
                            data=export_data,
                            file_name=f"construction_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.download_button(
                            label=f"Download {export_format.upper()}",
                            data=export_data,
                            file_name=f"construction_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            
            # Cleanup
            try:
                os.unlink(video_path)
            except:
                pass
        else:
            st.info("Upload a video and click 'Process Video' to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("**💡 Tip:** Start with the 'n' (nano) model for faster processing, upgrade to 'l' or 'x' for better accuracy")

if __name__ == "__main__":
    main()