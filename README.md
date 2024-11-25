# Camera Stabilizer

A Python-based video stabilization tool that uses L1-optimal camera path optimization to reduce camera shake and create smooth, professional-looking videos.

## Features

- L1-optimal path optimization for smooth camera motion
- Real-time preview with side-by-side comparison
- Adaptive border handling
- Progress tracking during processing
- Configurable smoothing parameters

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- CVXOPT (for L1 optimization)
- SciPy (for signal processing)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/camera_stabilizer.git
cd camera_stabilizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the stabilizer with a video file:
```bash
python camera_stabilizer.py input_video.mp4
```

The stabilized video will be saved as `stabilized_input_video.mp4` in the same directory.

## Parameters

You can adjust these parameters in the `VideoStabilizer` class:

- `smoothing_radius`: Controls the smoothness of camera motion (default: 31)
- `crop_ratio`: Controls how much to crop the borders (default: 0.1)

## How It Works

1. Detects feature points in each frame using goodFeaturesToTrack
2. Tracks these points between consecutive frames
3. Estimates frame-to-frame transformations
4. Optimizes the camera path using L1 optimization
5. Applies the smoothed transformations to create stabilized frames

## License

MIT License

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
