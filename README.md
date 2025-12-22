# 🎵 Dance Video Comparator

## Overview

The **Dance Video Comparator** is a professional-grade tool designed to compare two dance performance videos using human pose detection and joint-angle analysis. The application automatically aligns corresponding segments of the videos based on movement similarity and produces a synchronized, side-by-side comparison video with visual pose overlays.

This tool is well suited for choreography analysis, dance training, performance evaluation, and motion comparison research.

---

## ✨ Features

### 🧍 Pose Detection

* Powered by **Google MediaPipe Pose Landmarker** for accurate human pose estimation.

### ⏱ Automatic Temporal Alignment

* Automatically identifies matching sections between two videos.
* Uses cosine similarity of joint angles to compute frame offsets.

### 📐 Angle-Based Motion Analysis

* Extracts joint angles for arms, legs, torso, and shoulders.
* Robust against differences in camera position and dancer scale.

### 🎥 Video Processing

* Standardizes videos to **30 FPS**
* Optional horizontal mirroring for opposite-facing dancers
* Real-time pose landmark visualization

### 📺 Side-by-Side Comparison Output

* Generates an **MP4 comparison video**
* Color-coded poses:

  * **Green** — Reference video
  * **Orange** — Comparison video
* Displays alignment and playback metadata

### 🧑‍💻 Command-Line Interface

* Interactive prompts
* Step-by-step guidance throughout the pipeline

---

## 📋 Requirements

### Software

* **Python 3.8+**
* **FFmpeg**
* **MediaPipe Pose model** (`pose_landmarker.task`)

### Python Dependencies

All Python dependencies are listed in `requirements.txt`.

---

## 🔧 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/dance-video-comparator.git
cd dance-video-comparator
```

### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install FFmpeg

| Platform      | Command                                                |
| ------------- | ------------------------------------------------------ |
| Windows       | Download from [https://ffmpeg.org](https://ffmpeg.org) |
| macOS         | `brew install ffmpeg`                                  |
| Ubuntu/Debian | `sudo apt install ffmpeg`                              |

### 4️⃣ Download MediaPipe Model

Download **`pose_landmarker.task`** from:
[https://developers.google.com/mediapipe/solutions/vision/pose_landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

Place the file in the **project root directory**.

---

## 🚀 Usage

### 1️⃣ Add Input Videos

Place at least two video files (`.mp4`, `.avi`, `.mov`, `.mkv`) in the project directory.

### 2️⃣ Run the Application

```bash
python dance_comparator.py
```

### 3️⃣ Follow the Prompts

You will be prompted to:

* Select a reference video
* Select a comparison video
* Enable or disable mirroring
* Monitor processing progress

### 4️⃣ Output

A synchronized comparison video will be generated and saved as an **MP4 file**.

---

## 🧠 How It Works

### Processing Pipeline

1. **FPS Standardization**

   * Converts both videos to 30 FPS.

2. **Pose Detection**

   * MediaPipe detects body landmarks in sampled frames.

3. **Angle Extraction**

   * Calculates joint angles for limbs and torso.

4. **Alignment**

   * Sliding-window cosine similarity identifies the best temporal match.

5. **Output Rendering**

   * Videos are aligned, overlaid with poses, and rendered side by side.

---

## ⚙️ Configuration

### Change Sampling Rate

```python
angles1, _ = self.create_angle_arrays(video_path, frame_gap=10)
```

### Adjust Detection Confidence

```python
comparator = DanceVideoComparator(min_detection_confidence=0.5)
```

### Customize Output

Edit `save_videos_aligned()` to modify:

* Output resolution
* Frame rate
* Color schemes
* Overlay text and metadata

---

## 🐞 Troubleshooting

### ❌ Model File Not Found

**Solution:**
Ensure `pose_landmarker.task` is placed in the project directory.

---

### ❌ FFmpeg Not Found

**Solution:**
Install FFmpeg and ensure it is available in your system PATH:

```bash
ffmpeg -version
```

---

### 🐌 Slow Processing

**Solutions:**

* Reduce video resolution (720p recommended)
* Increase `frame_gap`
* Use shorter clips for testing

---

### 🧍 No Pose Detected

**Solutions:**

* Improve lighting
* Ensure dancer is fully visible
* Lower detection confidence (e.g., `0.3`)

---

### ⏱ Poor Alignment

**Solutions:**

* Enable mirroring if dancers face opposite directions
* Ensure similar choreography
* Verify pose landmarks are detected correctly

---

## 🚀 Performance Tips

* Pre-convert videos to **720p**
* Process **30–60 second** clips
* Increase sampling gap for long videos
* Close other CPU/GPU-intensive applications

---

## 🤝 Contributing

### Reporting Issues

Please include:

* Video format and resolution
* Error messages
* Steps to reproduce
* System specifications

### Code Contributions

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

* **MediaPipe** — Pose estimation framework by Google
* **OpenCV** — Computer vision and video processing
* **FFmpeg** — Multimedia processing and encoding
