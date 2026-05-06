<div align="center">

<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAA?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/BoxMOT-BoTSORT-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>

⚽ MiniMappaFootBall
Real-time football video analysis with player tracking, team classification, match statistics and tactical minimap.

Features - Demo - Installation - Usage - Architecture - Configuration - Roadmap

</div>

🎬 Demo
Output video with live panel and minimap overlay at 960×540 + 360px side panel

text
┌─────────────────────────────────────────┬──────────────────────┐
│                                         │  MATCH ANALYSIS 00:51│
│         VIDEO FRAME (960×540)           ├──────────────────────┤
│                                         │  BALL POSSESSION     │
│  #12  #7   #3                           │  80%  ─  20%         │
│  ●    ○    ●                            ├──────────────────────┤
│                                         │  ON FIELD   4 │ 9    │
│  ┌─────────────┐                        ├──────────────────────┤
│  │  MINIMAP    │                        │  PASSES     1 │ 1    │
│  │  [field]    │                        ├──────────────────────┤
│  └─────────────┘                        │  SPEED  15.6 │ 16.8  │
│                                         ├──────────────────────┤
│                                         │  DISTANCE  0.42 km   │
└─────────────────────────────────────────┴──────────────────────┘
✨ Features
🔍 Player & Ball Detection — YOLOv8 with custom soccana_best.pt model trained on football footage

🎯 Multi-Object Tracking — BoTSORT with camera motion compensation (CMC) for stable IDs across frames

👕 Team Classification — Automatic jersey color clustering in LAB color space with 30-frame history buffer for stability

📊 Live Match Statistics — Possession %, passes, average/max speed (km/h), distance covered (km) per team

🗺️ Tactical Minimap — Auto-calibrated homography from green field detection, real-time player positions projected to a 2D bird's-eye view

🎨 DAZN-style UI — Dark professional panel with animated progress bar, split possession bars, color-coded team cards

⚡ Threaded Pipeline — Inference worker runs in a separate thread; main thread handles I/O without blocking

📁 Project Structure
text
MiniMappaFootBall/
├── data/
│   └── raw/
│       └── input_vid.mp4          ← input video
├── models/
│   ├── soccana_best.pt            ← custom YOLO model for football
│   └── osnet_x0_25_msmt17.pt      ← ReID weights for BoTSORT
├── src/
│   ├── main.py                    ← entry point & rendering pipeline
│   ├── team_classifier.py         ← LAB-based jersey color classification
│   ├── stats_tracker.py           ← possession, passes, speed, distance
│   └── homography.py              ← minimap via automatic homography
├── .font_cache/
│   ├── Inter-Regular.otf
│   └── Inter-Bold.otf
├── team_colors.json               ← LAB centroids for both teams
├── output_football_analysis.mp4   ← generated output
└── .venv/
🔧 Installation
Prerequisites
Python 3.12

NVIDIA GPU with CUDA 12.1+ (tested on RTX 4060 Laptop)

PyTorch 2.5.1 with CUDA

Setup
powershell
# Clone the repository
git clone https://github.com/yourusername/MiniMappaFootBall.git
cd MiniMappaFootBall

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
Requirements
text
ultralytics>=8.0.0
boxmot>=10.0.0
supervision>=0.18.0
opencv-python>=4.9.0
Pillow>=10.0.0
scikit-learn>=1.4.0
numpy>=1.26.0
torch>=2.5.1
🚀 Usage
1. Calibrate Team Colors
Run the interactive color picker to sample jersey colors from your video:

powershell
python src/select_team_colors.py
Draw rectangles over jerseys of Team 0 (default)

Press TAB to switch to Team 1

Press ENTER to save → team_colors.json

2. Run Analysis
powershell
cd C:\path\to\MiniMappaFootBall
python src/main.py
Output video is saved to output_football_analysis.mp4.

⚙️ Configuration
All main parameters are at the top of src/main.py:

Parameter	Default	Description
MAX_SECONDS	60	Seconds of video to process (None = full video)
PLAYER_CONF	0.18	YOLO detection confidence threshold
INFER_SIZE	640	YOLO inference resolution (higher = more accurate, slower)
SAVE_VIDEO	True	Save output to .mp4
SHOW_PREVIEW	False	Show live OpenCV window during processing
FRAME_W / FRAME_H	960 / 540	Output frame resolution
PANEL_W	360	Width of the stats panel
🏗️ Architecture
text
Video Frame
     │
     ▼
preprocess_frame()          ← CLAHE contrast + sharpening
     │
     ├──► YOLO (players/refs) ──► BoTSORT tracker ──► track IDs
     │                                    │
     │                            TeamClassifier
     │                          (LAB color → team 0/1)
     │
     ├──► YOLO (ball) ──────────────────────────────► ball center
     │
     ▼
HomographyMapper
  ├── detect field corners (green mask → contour)
  ├── cv2.findHomography (pixel → metres)
  └── render_minimap() → 300×194px bird's-eye view
     │
     ▼
StatsTracker
  ├── possession (closest player to ball)
  ├── passes (team change after ≥6 frame streak)
  ├── speed (pixel displacement × METERS_PER_PIXEL × fps × 3.6)
  └── distance (cumulative pixel displacement)
     │
     ▼
build_panel() + build_canvas()
     │
     ▼
output_football_analysis.mp4
🗺️ Minimap Details
The minimap is computed via automatic homography calibration:

HSV green mask isolates the pitch from the background

Morphological cleaning removes noise (fans, billboards)

approxPolyDP extracts the 4 field corners; falls back to bounding rect with 3% margin

cv2.findHomography maps pixel coordinates → metric coordinates (105m × 68m)

Player foot positions are projected and rendered on a 300×194px field with alternating stripes

Recalibration runs every 30 frames to handle camera pan/zoom.

📊 Stats Engine
Metric	Method
Possession	Frame-by-frame: team of player closest to ball (threshold: 90px)
Recent possession	Rolling 5-second window
Passes	Team change event after ≥6 frames of continuous possession
Speed	px_displacement × 0.058 m/px × fps × 3.6 → km/h
Distance	Cumulative per-player displacement, filtered 0.5–60px/frame
Max speed	Per-team rolling maximum, capped at 60 km/h
🛣️ Roadmap
Multiple camera support with cross-view homography stitching

Formation detection (4-4-2, 4-3-3, etc.)

Heatmap generation per player/team

Export statistics to JSON/CSV

Web dashboard (Angular + Django)

Support for broadcast-quality 1080p/4K input

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

<div align="center">

Built with ❤️ using YOLOv8 · BoTSORT · OpenCV · Pillow · scikit-learn

</div>
