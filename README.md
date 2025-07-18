# ironsite-autolabeling

An automated construction video analysis system that uses GPT-4o vision to analyze construction site videos and automatically label time segments with specific construction activities. The system compares AI predictions against human ground truth labels to evaluate performance.

## Description

This tool processes construction videos to:
- **Automatically identify construction activities** using GPT-4o's computer vision capabilities
- **Create temporal segments** with start/end times and activity labels 
- **Compare predictions** against human-annotated ground truth data
- **Generate performance metrics** including mAP (mean Average Precision) and temporal IoU
- **Visualize results** with side-by-side comparison charts

### Supported Construction Activities

The system recognizes 40+ construction activities including:
- **Pipe Installation**: Copper, Cast Iron, PVC, Steel, Hangers & Supports
- **Framing & Drywall**: Metal/Wood Framing, Drywall Hang/Finish, Insulation
- **General Activities**: Discussion, Planning, Transport, Safety, Cleanup
- **Non-Productive Time**: Phone usage, Waiting/Idle, Breaks

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key with GPT-4o access
- CSV file with video data (see format below)

### Installation

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd ironsite-autolabeling
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure API key**:
   - Open `gpt4o_v2.0.1.py`
   - Replace `"YOUR_API_KEY"` with your actual OpenAI API key (line 856)

3. **Prepare video data**:
   Create `ironsite_videos.csv` with columns:
   - `Video ID`: Unique identifier
   - `Video URL`: Direct video file URL  
   - `Ground Truth Segments`: Format: `"0:00-2:05|Hangers & Supports,2:06-4:54|Cast Iron"`

### Running the Analysis

```bash
python gpt4o_v2.0.1.py
```

The system will:
1. Download videos from URLs
2. Extract frames for GPT-4o analysis
3. Generate activity predictions with confidence scores
4. Compare against ground truth labels
5. Save results and visualizations to `results/` folder

### Output

- **`results.json`**: Complete evaluation metrics and predictions
- **`{video_id}_comparison.png`**: Visual comparison charts
- **Performance metrics**: mAP@0.5, temporal IoU, precision/recall

### CSV Format Example

```csv
Video ID,Video URL,Ground Truth Segments
1,https://example.com/video1.mp4,"0:00-2:05|Hangers & Supports,2:06-4:54|Cast Iron"
2,https://example.com/video2.mp4,"0:00-1:30|Discussion,1:31-3:45|Copper"
```

## Rate Limiting

The system includes built-in rate limiting (30 requests/minute) to comply with OpenAI API limits. Processing time scales with video count and duration.
