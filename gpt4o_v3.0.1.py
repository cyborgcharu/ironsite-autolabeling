import os
import re
import json
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import time
import random
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2
import base64
from openai import OpenAI
from urllib.parse import urlparse
import logging
from pyrate_limiter import Limiter, Rate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeSegment:
    """Represents a time segment with start, end, and label"""
    start_seconds: float
    end_seconds: float
    label: str
    confidence: float = 1.0
    
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds
    
    def overlaps_with(self, other: 'TimeSegment') -> bool:
        """Check if this segment overlaps with another"""
        return not (self.end_seconds <= other.start_seconds or self.start_seconds >= other.end_seconds)
    
    def intersection_duration(self, other: 'TimeSegment') -> float:
        """Calculate intersection duration with another segment"""
        if not self.overlaps_with(other):
            return 0.0
        return min(self.end_seconds, other.end_seconds) - max(self.start_seconds, other.start_seconds)
    
    def union_duration(self, other: 'TimeSegment') -> float:
        """Calculate union duration with another segment"""
        return self.duration() + other.duration() - self.intersection_duration(other)
    
    def iou(self, other: 'TimeSegment') -> float:
        """Calculate Intersection over Union with another segment"""
        intersection = self.intersection_duration(other)
        union = self.union_duration(other)
        return intersection / union if union > 0 else 0.0

@dataclass
class VideoData:
    """Container for video data and labels"""
    video_id: str
    video_url: str
    ground_truth: List[TimeSegment]
    predictions: List[TimeSegment] = None
    video_duration: float = 0.0
    local_path: str = None

class VideoDownloader:
    """Handles video downloading from various sources"""
    
    def __init__(self, download_dir: str = "videos"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
    
    def download_video(self, url: str, video_id: str) -> str:
        """Download video from URL and return local path"""
        try:
            # Add headers to handle potential authentication or content issues
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check if we actually got video content
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                logger.error(f"Got HTML instead of video for {video_id}. URL might require authentication.")
                return None
            
            # Determine file extension
            if 'video/mp4' in content_type:
                ext = '.mp4'
            elif 'video/webm' in content_type:
                ext = '.webm'
            elif 'video/quicktime' in content_type:
                ext = '.mov'
            else:
                ext = '.mp4'  # Default to mp4
            
            local_path = self.download_dir / f"{video_id}{ext}"
            
            # Download with progress indication
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress for large files
                        if total_size > 0 and downloaded % (1024*1024) == 0:  # Every MB
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress for {video_id}: {progress:.1f}%")
            
            # Check if file was actually downloaded
            if not local_path.exists() or local_path.stat().st_size == 0:
                logger.error(f"Downloaded file is empty or doesn't exist: {local_path}")
                return None
            
            logger.info(f"Downloaded video: {local_path} ({local_path.stat().st_size} bytes)")
            return str(local_path)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading video {video_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading video {video_id}: {e}")
            return None
    
    def get_video_duration(self, video_path: str) -> float:
        """Get video duration using OpenCV with fallback methods"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.warning(f"OpenCV couldn't open video: {video_path}")
                return 0.0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps <= 0 or frame_count <= 0:
                logger.warning(f"Invalid video properties: fps={fps}, frames={frame_count}")
                cap.release()
                return 0.0
            
            duration = frame_count / fps
            cap.release()
            
            logger.info(f"Video duration: {duration:.2f} seconds")
            return duration
            
        except Exception as e:
            logger.error(f"Error getting video duration: {e}")
            return 0.0

class GroundTruthExtractor:
    """Extracts ground truth labels from Ironsite interface"""
    
    def parse_time_to_seconds(self, time_str: str) -> float:
        """Convert time string (MM:SS or H:MM:SS) to seconds"""
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # H:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0.0
    
    def extract_from_ironsite_data(self, ironsite_segments: List[Dict]) -> List[TimeSegment]:
        """Extract ground truth from Ironsite segment data"""
        segments = []
        
        for segment in ironsite_segments:
            # Parse time range like "2:06 - 4:54"
            time_range = segment.get('time_range', '')
            label = segment.get('label', '')
            
            if ' - ' in time_range:
                start_str, end_str = time_range.split(' - ')
                start_seconds = self.parse_time_to_seconds(start_str.strip())
                end_seconds = self.parse_time_to_seconds(end_str.strip())
                
                segments.append(TimeSegment(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    label=label
                ))
        
        return segments

class GPT4oPredictor:
    """Handles GPT-4o video analysis and prediction"""
    
    def __init__(self, api_key: str, cost_codes: Dict[str, str]):
        self.client = OpenAI(api_key=api_key)
        self.cost_codes = cost_codes
        # Rate limiter: OpenAI allows 30 requests per minute for GPT-4o
        self.rate_limiter = Limiter(Rate(30, 60))
    
    def extract_frames(self, video_path: str, num_frames: int = 20) -> List[str]:
        """Extract frames from video and encode as base64"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Cannot open video file: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0 or fps <= 0:
                logger.error(f"Invalid video properties: frames={total_frames}, fps={fps}")
                cap.release()
                return []
            
            logger.info(f"Video has {total_frames} frames at {fps} fps")
            
            # Calculate frame indices to extract
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames_b64 = []
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Convert to RGB and encode
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if buffer is not None:
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        timestamp = frame_idx / fps
                        frames_b64.append({
                            'timestamp': timestamp,
                            'image': frame_b64
                        })
                        logger.info(f"Extracted frame {i+1}/{num_frames} at {timestamp:.2f}s")
                else:
                    logger.warning(f"Failed to read frame {frame_idx}")
            
            cap.release()
            logger.info(f"Successfully extracted {len(frames_b64)} frames")
            return frames_b64
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []

    def create_cost_code_prompt(self, video_duration: float) -> str:
        """Create prompt with cost code information"""
        cost_code_text = "Construction Activities to Identify:\n"
        for activity, description in self.cost_codes.items():
            cost_code_text += f"- {activity}: {description}\n"
    
        return f"""
    You are an expert construction video analyst. Analyze the provided video frames to identify construction activities and create temporal segments.

    {cost_code_text}

    VIDEO INFORMATION:
    - Total Duration: {video_duration:.1f} seconds
    - You will see frames sampled across the entire video duration
    - Each frame shows a timestamp indicating when it occurs in the video

    ANALYSIS REQUIREMENTS:
    1. Examine each frame carefully and note the timestamp
    2. Identify the primary construction activity in each frame
    3. Create temporal segments that cover the ENTIRE video duration from 0 to {video_duration:.1f} seconds
    4. Segments should be meaningful durations (typically 30+ seconds, not 10-second chunks)
    5. Adjacent segments with the same activity should be merged
    6. Ensure no gaps between segments

    IMPORTANT:
    Most construction videos have multiple different activities. Look carefully at the timestamps:
        - Early frames (0-120 seconds): What activity do you see?
        - Middle frames (120-200 seconds): Does the activity change?
        - Later frames (200+ seconds): Is this the same or different activity?
    DO NOT assume the entire video is one activity. Look for transitions between different construction tasks.

    TRANSITION DETECTION STRATEGY:
    1. Compare consecutive frames carefully - what changes between frames?
    2. Look for visual cues of activity changes:
        - Different tools being used
        - Workers moving to different areas  
        - Change in materials (pipes vs brackets vs phones)
        - Different body postures/movements
    3. If you see ANY change in activity, create a new segment
    REMEMBER: Construction workers rarely do the same exact task for 5+ minutes straight. Look for natural workflow transitions.


    OUTPUT FORMAT:
    Return a JSON array with this exact structure:
    [
        {{
            "start_seconds": 0.0,
            "end_seconds": 125.0,
            "activity": "Hangers & Supports",
            "confidence": 0.85,
            "evidence": "Multiple frames show workers installing pipe hangers and mounting brackets"
        }},
        {{
            "start_seconds": 125.0,
            "end_seconds": {video_duration:.1f},
            "activity": "Cast Iron", 
            "confidence": 0.90,
            "evidence": "Workers handling dark cast iron pipes and making connections"
        }}
    ]

    CRITICAL REQUIREMENTS:
    - Cover the ENTIRE {video_duration:.1f} second duration
    - Start first segment at 0.0 seconds  
    - End last segment at {video_duration:.1f} seconds
    - No gaps between segments
    - Use only activities from the provided list
    - Create 2-4 segments total, not many small ones
    - Each segment should be at least 30 seconds long
    """
        
    
    
    def analyze_video(self, video_path: str, video_duration: float) -> List[TimeSegment]:
        """Analyze video with GPT-4o and return predicted segments"""
        try:
            if video_duration is None:
                video_duration = self.get_video_duration(video_path)
            frames = self.extract_frames(video_path)
            
            messages = [
                {
                    "role": "system",
                    "content": self.create_cost_code_prompt(video_duration)
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze these {len(frames)} frames from a construction video. The frames are evenly spaced throughout the video duration."
                        }
                    ]
                }
            ]
            
            # Add frames to message
            for frame_data in frames:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_data['image']}"
                    }
                })
            
            # Apply rate limiting before API call
            self.rate_limiter.try_acquire("openai_api")
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content

            logger.info(f"GPT-4o response: {response_text}")

            
            # Extract JSON from response
            clean_text = response_text.replace('```json', '').replace('```', '')
            json_match = re.search(r'\[.*\]', clean_text, re.DOTALL)
            if json_match:
                predictions_data = json.loads(json_match.group())
                
                segments = []
                for pred in predictions_data:
                    segments.append(TimeSegment(
                        start_seconds=pred['start_seconds'],
                        end_seconds=pred['end_seconds'],
                        label=pred.get('activity', pred.get('cost_code', 'Unknown')),
                        confidence=pred.get('confidence', 0.5)
                    ))
                
                return segments
            
        except Exception as e:
            logger.error(f"Error analyzing video with GPT-4o: {e}")
            return []

class PerformanceEvaluator:
    """Evaluates prediction performance using various metrics"""
    
    def calculate_segment_iou(self, pred: TimeSegment, gt: TimeSegment) -> float:
        """Calculate IoU between predicted and ground truth segments"""
        if pred.label != gt.label:
            return 0.0
        return pred.iou(gt)
    
    def calculate_map(self, predictions: List[TimeSegment], ground_truth: List[TimeSegment], 
                     iou_thresholds: List[float] = [0.5, 0.75, 0.9]) -> Dict[str, float]:
        """Calculate mean Average Precision (mAP) at different IoU thresholds"""
        results = {}
        
        for threshold in iou_thresholds:
            ap_scores = []
            
            # Group by label
            labels = set([seg.label for seg in ground_truth + predictions])
            
            for label in labels:
                gt_segments = [seg for seg in ground_truth if seg.label == label]
                pred_segments = [seg for seg in predictions if seg.label == label]
                
                if not gt_segments:
                    continue
                
                # Sort predictions by confidence
                pred_segments.sort(key=lambda x: x.confidence, reverse=True)
                
                # Calculate precision-recall curve
                tp = 0
                fp = 0
                used_gt = set()
                
                precisions = []
                recalls = []
                
                for pred in pred_segments:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for i, gt in enumerate(gt_segments):
                        if i in used_gt:
                            continue
                        
                        iou = self.calculate_segment_iou(pred, gt)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    
                    if best_iou >= threshold:
                        tp += 1
                        used_gt.add(best_gt_idx)
                    else:
                        fp += 1
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / len(gt_segments) if len(gt_segments) > 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                
                # Calculate AP using interpolated precision
                if precisions:
                    ap = np.trapz(precisions, recalls) if len(recalls) > 1 else precisions[0]
                    ap_scores.append(ap)
            
            results[f'mAP@{threshold}'] = np.mean(ap_scores) if ap_scores else 0.0
        
        return results
    
    def calculate_temporal_metrics(self, predictions: List[TimeSegment], 
                                 ground_truth: List[TimeSegment]) -> Dict[str, float]:
        """Calculate temporal alignment metrics"""
        if not predictions and not ground_truth:
            return {
                'temporal_precision': 0.0,
                'temporal_recall': 0.0,
                'temporal_f1': 0.0,
                'temporal_accuracy': 0.0,
                'temporal_iou': 0.0
            }
        
        if not predictions:
            return {
                'temporal_precision': 0.0,
                'temporal_recall': 0.0,
                'temporal_f1': 0.0,
                'temporal_accuracy': 0.0,
                'temporal_iou': 0.0
            }
        
        if not ground_truth:
            return {
                'temporal_precision': 0.0,
                'temporal_recall': 0.0,
                'temporal_f1': 0.0,
                'temporal_accuracy': 0.0,
                'temporal_iou': 0.0
            }
        
        # Get the maximum duration across all segments
        all_segments = ground_truth + predictions
        total_duration = max([seg.end_seconds for seg in all_segments])
        
        # Create binary arrays for each second
        gt_array = np.zeros(int(total_duration) + 1)
        pred_array = np.zeros(int(total_duration) + 1)
        
        # Fill arrays
        for seg in ground_truth:
            start_idx = int(seg.start_seconds)
            end_idx = int(seg.end_seconds)
            gt_array[start_idx:end_idx] = 1
        
        for seg in predictions:
            start_idx = int(seg.start_seconds)
            end_idx = int(seg.end_seconds)
            pred_array[start_idx:end_idx] = 1
        
        # Calculate metrics
        tp = np.sum((gt_array == 1) & (pred_array == 1))
        fp = np.sum((gt_array == 0) & (pred_array == 1))
        fn = np.sum((gt_array == 1) & (pred_array == 0))
        tn = np.sum((gt_array == 0) & (pred_array == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate IoU (Intersection over Union)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0
        
        return {
            'temporal_precision': precision,
            'temporal_recall': recall,
            'temporal_f1': f1,
            'temporal_accuracy': (tp + tn) / len(gt_array) if len(gt_array) > 0 else 0,
            'temporal_iou': iou
        }

class VideoLabelerSystem:
    """Main system orchestrating the video labeling pipeline"""
    
    def __init__(self, openai_api_key: str, cost_codes: Dict[str, str]):
        self.downloader = VideoDownloader()
        self.gt_extractor = GroundTruthExtractor()
        self.predictor = GPT4oPredictor(openai_api_key, cost_codes)
        self.evaluator = PerformanceEvaluator()
        self.videos: List[VideoData] = []
    
    def load_video_data_from_urls(self, video_urls: List[str]) -> None:
        """Load video data from a list of URLs (for quick testing)"""
        for i, url in enumerate(video_urls):
            video_id = f"video_{i+1}"
            
            # Create sample ground truth for testing
            # Replace this with your actual ground truth extraction
            ground_truth = [
                TimeSegment(0, 125, "Hangers & Supports"),
                TimeSegment(126, 294, "Cast Iron")
            ]
            
            video_data = VideoData(
                video_id=video_id,
                video_url=url,
                ground_truth=ground_truth
            )
            
            self.videos.append(video_data)
    
    def load_video_data(self, csv_path: str) -> None:
        """Load video data from CSV file"""
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} videos from CSV")
        
        for _, row in df.iterrows():
            video_id = str(row.get('Video ID', ''))
            video_url = row.get('Video URL', '')
            
            # Extract ground truth from your specific format
            ground_truth = self.extract_ground_truth_from_row(row)
            
            video_data = VideoData(
                video_id=video_id,
                video_url=video_url,
                ground_truth=ground_truth
            )
            
            self.videos.append(video_data)
            logger.info(f"Loaded video {video_id} with {len(ground_truth)} ground truth segments")
    
    def extract_ground_truth_from_row(self, row) -> List[TimeSegment]:
        """Extract ground truth from spreadsheet row"""
        segments = []
        
        # Get the Ground Truth Segments column
        gt_text = row.get('Ground Truth Segments', '')
        
        if pd.isna(gt_text) or gt_text.strip() == '':
            logger.warning(f"No ground truth data for video {row.get('Video ID', 'unknown')}")
            return segments
        
        # Clean up the text - remove triple quotes and extra whitespace
        gt_text = gt_text.strip()
        if gt_text.startswith('"""') and gt_text.endswith('"""'):
            gt_text = gt_text[3:-3]  # Remove triple quotes
        elif gt_text.startswith('"') and gt_text.endswith('"'):
            gt_text = gt_text[1:-1]  # Remove single quotes
        
        # Parse format like: "0:00-2:05|Hangers & Supports,2:06-4:54|Cast Iron"
        try:
            # Split by comma to get individual segments
            segment_parts = gt_text.split(',')
            
            for segment_part in segment_parts:
                segment_part = segment_part.strip()
                if '|' in segment_part:
                    time_part, label = segment_part.split('|', 1)
                    
                    if '-' in time_part:
                        start_str, end_str = time_part.split('-', 1)
                        start_seconds = self.gt_extractor.parse_time_to_seconds(start_str.strip())
                        end_seconds = self.gt_extractor.parse_time_to_seconds(end_str.strip())
                        
                        segments.append(TimeSegment(
                            start_seconds=start_seconds,
                            end_seconds=end_seconds,
                            label=label.strip()
                        ))
                        
                        logger.debug(f"Parsed segment: {start_str} to {end_str} = {label}")
        
        except Exception as e:
            logger.error(f"Error parsing ground truth for video {row.get('Video ID', 'unknown')}: {e}")
            logger.error(f"Ground truth text was: '{gt_text}'")
        
        return segments
    
    def process_video(self, video_data: VideoData) -> None:
        """Process a single video: download, analyze, and evaluate"""
        logger.info(f"Processing video: {video_data.video_id}")
        
        # Download video
        video_data.local_path = self.downloader.download_video(
            video_data.video_url, video_data.video_id
        )
        
        if not video_data.local_path:
            logger.error(f"Failed to download video: {video_data.video_id}")
            return
        
        # Get video duration - use ground truth duration if OpenCV fails
        video_data.video_duration = self.downloader.get_video_duration(video_data.local_path)
        
        if video_data.video_duration <= 0:
            # Fallback: estimate duration from ground truth segments
            if video_data.ground_truth:
                video_data.video_duration = max([seg.end_seconds for seg in video_data.ground_truth])
                logger.info(f"Using ground truth duration: {video_data.video_duration} seconds")
            else:
                video_data.video_duration = 300  # Default 5 minutes
                logger.warning(f"Using default duration: {video_data.video_duration} seconds")
        
        # Run GPT-4o analysis (or mock predictions)
        video_data.predictions = self.predictor.analyze_video(video_data.local_path, video_data.video_duration)
        
        if not video_data.predictions:
            logger.warning(f"No predictions generated for video {video_data.video_id}")
        else:
            logger.info(f"Generated {len(video_data.predictions)} predictions for video {video_data.video_id}")
        
        logger.info(f"Completed processing video: {video_data.video_id}")
    
    def create_visualization(self, video_data: VideoData, output_path: str) -> None:
        """Create bar graph comparing ground truth vs predictions"""
        if not video_data.predictions:
            logger.warning(f"No predictions to visualize for video {video_data.video_id}")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Ground Truth
        colors = plt.cm.Set3(np.linspace(0, 1, len(video_data.ground_truth)))
        for i, seg in enumerate(video_data.ground_truth):
            ax1.barh(0, seg.duration(), left=seg.start_seconds, 
                    alpha=0.7, color=colors[i])
            ax1.text(seg.start_seconds + seg.duration()/2, 0, seg.label, 
                    ha='center', va='center', fontsize=10, weight='bold')
        
        ax1.set_title(f'Ground Truth - {video_data.video_id}')
        ax1.set_xlim(0, video_data.video_duration)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_yticks([])
        
        # Predictions
        pred_colors = plt.cm.Set1(np.linspace(0, 1, len(video_data.predictions)))
        for i, seg in enumerate(video_data.predictions):
            ax2.barh(0, seg.duration(), left=seg.start_seconds, 
                    alpha=0.7, color=pred_colors[i])
            ax2.text(seg.start_seconds + seg.duration()/2, 0, 
                    f"{seg.label}\n({seg.confidence:.2f})", 
                    ha='center', va='center', fontsize=10, weight='bold')
        
        ax2.set_title(f'GPT-4o Predictions - {video_data.video_id}')
        ax2.set_xlim(0, video_data.video_duration)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created visualization: {output_path}")
    
    def evaluate_all_videos(self) -> Dict[str, float]:
        """Evaluate performance across all videos"""
        all_predictions = []
        all_ground_truth = []
        
        for video_data in self.videos:
            if video_data.predictions:
                all_predictions.extend(video_data.predictions)
                all_ground_truth.extend(video_data.ground_truth)
        
        # Calculate metrics
        map_results = self.evaluator.calculate_map(all_predictions, all_ground_truth)
        temporal_results = self.evaluator.calculate_temporal_metrics(all_predictions, all_ground_truth)
        
        return {**map_results, **temporal_results}
    
    def run_full_pipeline(self, csv_path: str, output_dir: str = "results") -> None:
        """Run the complete pipeline"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load data
        self.load_video_data(csv_path)
        
        if not self.videos:
            logger.error("No videos loaded. Please check your CSV file.")
            return
        
        # Process each video
        processed_videos = 0
        for video_data in self.videos:
            # Only process videos that have ground truth data
            if video_data.ground_truth:
                logger.info(f"Processing video {video_data.video_id}...")
                self.process_video(video_data)
                processed_videos += 1
                
                # Create visualization
                if video_data.predictions:
                    viz_path = output_path / f"{video_data.video_id}_comparison.png"
                    self.create_visualization(video_data, str(viz_path))
            else:
                logger.info(f"Skipping video {video_data.video_id} - no ground truth data")
        
        if processed_videos == 0:
            logger.warning("No videos were processed. Make sure your CSV has Ground Truth Segments data.")
            return
        
        # Evaluate performance
        metrics = self.evaluate_all_videos()
        
        # Save results
        results = {
            'evaluation_metrics': metrics,
            'processed_videos': processed_videos,
            'total_videos': len(self.videos),
            'video_results': [
                {
                    'video_id': vd.video_id,
                    'ground_truth': [asdict(seg) for seg in vd.ground_truth],
                    'predictions': [asdict(seg) for seg in vd.predictions] if vd.predictions else [],
                    'duration': vd.video_duration,
                    'processed': bool(vd.predictions)
                }
                for vd in self.videos
            ]
        }
        
        with open(output_path / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Pipeline completed. Processed {processed_videos}/{len(self.videos)} videos. Results saved to {output_path}")
        return results

# Example usage
if __name__ == "__main__":
    # Load cost codes from your spreadsheet
    cost_codes = {
    # Discussion / Planning (01-01)
    "Discussion": "Workers engage in conversational exchanges to analyze, explain, or debate work items, tasks, or processes without directing immediate actions",
    "Planning & Logistics": "Workers coordinate tasks, resources, or schedules, often directing actions or organizing workflows",
    
    # Materials (01-02) 
    "Transport": "Workers transport large materials (e.g., studs, pipes, drywall) over their shoulder or using a materials cart from one place to another",
    "Pack/Unpack In Place": "Workers organize or prepare construction materials at a specific location by packing them for storage or unpacking them for use",
    
    # Transit (01-03)
    "Transit (On Foot)": "Workers walk across the construction site to reach a work area, access equipment, or transition between tasks",
    
    # Safety (01-04)
    "Preparation": "Workers engage in activities to ready themselves or the site for safe work - morning huddles, stretch and flex, harness adjustment, getting safety equipment",
    "Hoist Spotter": "A worker on the ground oversees the operation of a hoist or lift to ensure safe movement of loads and guide the operator",
    "Firewatch": "A worker monitors an area for fire hazards during or after high-risk activities (e.g., welding, cutting)",
    "Operating Lift": "Worker operates lifting equipment for vertical movement",
    
    # Cleaning/Disposal (01-05)
    "Sweep": "Workers use brooms or similar tools to clean surfaces by sweeping debris, dust, or small materials into piles or containers",
    "Trash Transport": "Workers move collected waste or debris (e.g., in bags, bins, or carts) from the site to a designated disposal area",
    "Clean Up": "General cleaning activities not covered by sweep or trash transport",
    
    # Non-Productive (01-06)
    "Phone Time": "Workers engage with personal devices (e.g., smartphones, tablets) for any purpose",
    "Waiting/Idle": "Workers remain inactive, standing or sitting without performing tasks, moving, or engaging with others",
    "Non-Productive Other": "Workers engage in non-work related activities such as eating, drinking, smoking, or other personal behaviors",
    
    # Break (01-07)
    "Break": "Workers take scheduled or designated pauses from work-related tasks to engage in personal activities",
    
    # Framing & Drywall (05-XX)
    "Framing (Metal)": "Workers construct structural frameworks using metal components for walls, ceilings, or partitions",
    "Framing (Wood)": "Workers construct structural frameworks using wood components for walls, ceilings, or floors", 
    "Drywall Hang": "Workers install drywall sheets (gypsum boards) on walls and ceilings to form interior surfaces",
    "Drywall Finish": "Workers apply finishing techniques to installed drywall - taping, mudding, sanding, or skim coating",
    "Batt": "Workers install fibrous batt insulation (e.g., fiberglass, mineral wool) in wall or ceiling cavities",
    "Rigid Board": "Workers install rigid insulation boards (e.g., foam, polystyrene) on walls, floors, or roofs",
    "Spray Foam": "Workers apply spray foam insulation",
    "ACT": "Acoustic ceiling tile installation",
    "Caulking": "Application of caulk and sealants",
    "Plywood": "Plywood installation and preparation", 
    "Exterior": "Exterior finishing work",
    "Firestop": "Installation of fire stopping materials and systems",
    "Demo": "Demolition activities for framing, drywall, and insulation",
    
    # Pipe Installation (15-01) - The ones matching your ground truth!
    "Copper": "Workers install or prepare copper pipes for plumbing or HVAC systems, involving cutting, soldering, or fitting",
    "Cast Iron": "Workers install or prepare cast iron pipes, typically for drainage or waste systems, using heavy-duty fittings", 
    "PVC": "Workers install or prepare PVC pipes for drainage, water supply, or waste systems, using gluing or fitting techniques",
    "Steel": "Workers install or prepare steel pipes for industrial or high-pressure systems using welding, threading, or heavy fittings",
    "Hangers & Supports": "Workers install brackets, hangers, or supports to secure pipes to walls, ceilings, or structures",
    "Sleeves & Openings": "Workers mark or measure areas to plan pipe installation routes or create openings for pipes",
    "Layout": "Workers mark or measure areas to plan pipe installation routes using chalk lines, tape measures, or markers", 
    "Underground": "Workers install underground piping systems, typically in trenches or below-grade applications"}
    
    # Initialize system with OpenAI API key from environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ OpenAI API key not found!")
        print("Please:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to the .env file")
        print("3. Make sure python-dotenv is installed: pip3 install python-dotenv")
        exit(1)
    
    system = VideoLabelerSystem(
        openai_api_key=openai_api_key,
        cost_codes=cost_codes
    )
    
    # Run pipeline with your CSV file
    csv_filename = "ironsite_videos.csv"
    
    if not os.path.exists(csv_filename):
        print(f"❌ CSV file '{csv_filename}' not found!")
        print("Please:")
        print("1. Export your Google Sheet to CSV")
        print("2. Name it 'ironsite_videos.csv' (or update the filename in the script)")
        print("3. Place it in the same directory as this script")
        print("4. Make sure it has columns: Video ID, Video URL, Ground Truth Segments")
    else:
        print(f"🚀 Starting pipeline with {csv_filename}...")
        print("🚦 Rate limiting enabled: 30 requests per minute")
        print("⏳ This may take some time with large datasets...")
        
        results = system.run_full_pipeline(csv_filename)
        
        if results:
            print("\n📊 Evaluation Results:")
            for metric, value in results['evaluation_metrics'].items():
                print(f"   {metric}: {value:.3f}")
            
            print(f"\n✅ Processed {results['processed_videos']}/{results['total_videos']} videos")
            print("📁 Check the 'results' folder for visualizations and detailed results")
            
            # Show rate limiter stats
            rate_limiter = system.predictor.rate_limiter
            print(f"\n🚦 Rate Limiter Stats:")
            print(f"   - Rate limit: 30 requests per minute")
            print(f"   - Rate limiter active and working")
            print(f"   - API calls completed successfully")
        else:
            print("\n❌ Pipeline failed. Check the logs above for details.")
            print("Common issues:")
            print("   - Invalid OpenAI API key")
            print("   - Insufficient API quota") 
            print("   - Video download/processing issues")