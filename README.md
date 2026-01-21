# SpatialAudioTracker 🎧

**SpatialAudioTracker** is an automated tool designed to extract audio from video, identify sound-emitting sources via computer vision, and dynamically re-position those sources in the stereo field based on their visual movement.



## 📖 Overview

This project bridges the gap between visual movement and static audio. By tracking objects in a video frame, the system calculates the appropriate panning ($L/R$ balance) and depth, ensuring that what you see on the left of the screen is heard in the left ear of your headphones.

### Key Features
* **Source Separation:** Isolates audio layers from complex video files.
* **Object Tracking:** Uses Computer Vision to follow sound sources across the screen.
* **Automated Panning:** Real-time calculation of stereo positioning based on $X$-axis coordinates.
* **Immersive Output:** Optimized for headphone users to provide a spatialized listening experience.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have **Python 3.8+** and **FFmpeg** installed on your system.

### Installation
We provide a script to handle all necessary dependencies (OpenCV, Pydub, NumPy, etc.) automatically.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/houslast/Audioseparatorstereo.git](https://github.com/houslast/Audioseparatorstereo.git)
    cd Audioseparatorstereo
    ```

2.  **Install Dependencies:**
    Run the following command to set up your environment:
    ```bash
    python install_deps.bat
    ```

---

## 🛠 Usage

### Starting the Server
The project includes a web-based interface and API backend to process your videos. To launch it, run:

```bash
python run_server.bat
