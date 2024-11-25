import cv2
import numpy as np
from typing import Tuple, List, Optional
import sys
import os
from scipy.signal import medfilt
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False  # Disable solver output

class VideoStabilizer:
    def __init__(self, smoothing_radius: int = 31, crop_ratio: float = 0.1):
        # Make sure smoothing_radius is odd
        self.smoothing_radius = smoothing_radius + 1 if smoothing_radius % 2 == 0 else smoothing_radius
        self.crop_ratio = crop_ratio
        self.transforms = []
        self.frame_buffer = []
        self.smoothed_transforms = None
        
    def _detect_features(self, frame_gray: np.ndarray) -> np.ndarray:
        """Detect good features to track."""
        return cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )

    def _compute_transform(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> np.ndarray:
        """Compute transformation matrix between two sets of points."""
        transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC,
                                              ransacReprojThreshold=3)[0]
        if transform is None:
            return np.eye(2, 3)
        return transform

    def _extract_transform_params(self, transform: np.ndarray) -> np.ndarray:
        """Extract dx, dy, da from transform matrix."""
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
        return np.array([dx, dy, da])

    def _create_transform_matrix(self, params: np.ndarray) -> np.ndarray:
        """Create transform matrix from dx, dy, da."""
        dx, dy, da = params
        transform = np.zeros((2, 3))
        transform[0, 0] = np.cos(da)
        transform[0, 1] = -np.sin(da)
        transform[1, 0] = np.sin(da)
        transform[1, 1] = np.cos(da)
        transform[0, 2] = dx
        transform[1, 2] = dy
        return transform

    def _optimize_path(self, transforms_params: np.ndarray) -> np.ndarray:
        """Apply L1-optimal path optimization."""
        n = len(transforms_params)
        if n < 3:
            return transforms_params

        # Median filter for initial smoothing
        smoothed = np.zeros_like(transforms_params)
        for i in range(3):  # For each parameter (dx, dy, da)
            smoothed[:, i] = medfilt(transforms_params[:, i], kernel_size=min(self.smoothing_radius, n))

        # Set up optimization problem for each parameter
        optimized = np.zeros_like(transforms_params)
        for i in range(3):
            # Original path
            path = transforms_params[:, i]
            
            # Set up quadratic program matrices
            P = matrix(2.0 * np.eye(n))
            q = matrix(-2.0 * path)
            
            # First derivative constraints
            G1 = np.zeros((n-1, n))
            for j in range(n-1):
                G1[j, j:j+2] = [1, -1]
            
            # Second derivative constraints
            G2 = np.zeros((n-2, n))
            for j in range(n-2):
                G2[j, j:j+3] = [1, -2, 1]
            
            # Combine constraints
            G = matrix(np.vstack([G1, G2]))
            h = matrix(np.zeros(2*n-3))
            
            # Solve optimization problem
            sol = solvers.qp(P, q, G, h)
            if sol['status'] == 'optimal':
                optimized[:, i] = np.array(sol['x']).flatten()
            else:
                optimized[:, i] = smoothed[:, i]

        return optimized

    def precompute_stabilization(self, frames: List[np.ndarray]) -> None:
        """Precompute stabilization transforms for all frames."""
        n_frames = len(frames)
        self.transforms = []
        transforms_params = np.zeros((n_frames-1, 3))  # dx, dy, da for each frame transition

        # Compute frame-to-frame transforms
        for i in range(n_frames - 1):
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)

            prev_pts = self._detect_features(prev_gray)
            if prev_pts is None:
                transform = np.eye(2, 3)
            else:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
                good_prev = prev_pts[status.ravel() == 1]
                good_curr = curr_pts[status.ravel() == 1]

                if len(good_prev) < 10:
                    transform = np.eye(2, 3)
                else:
                    transform = self._compute_transform(good_prev, good_curr)

            transforms_params[i] = self._extract_transform_params(transform)
            self.transforms.append(transform)

        # Optimize camera path
        optimized_params = self._optimize_path(transforms_params)

        # Convert optimized parameters back to transformation matrices
        self.smoothed_transforms = [self._create_transform_matrix(params) for params in optimized_params]

    def stabilize_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Stabilize a frame using precomputed transforms."""
        if frame_idx >= len(self.smoothed_transforms):
            return frame

        # Apply transform
        h, w = frame.shape[:2]
        transform = self.smoothed_transforms[frame_idx]
        stabilized = cv2.warpAffine(frame, transform, (w, h))

        # Crop borders
        crop_x = int(w * self.crop_ratio)
        crop_y = int(h * self.crop_ratio)
        stabilized = stabilized[crop_y:h-crop_y, crop_x:w-crop_x]
        
        # Resize back to original size
        stabilized = cv2.resize(stabilized, (w, h))

        return stabilized

def main():
    if len(sys.argv) != 2:
        print("Usage: python camera_stabilizer.py video_file_path")
        return

    # Read input video
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    output_path = 'stabilized_' + os.path.basename(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read all frames
    print("Reading frames...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Initialize stabilizer
    print("Computing stabilization...")
    stabilizer = VideoStabilizer()
    stabilizer.precompute_stabilization(frames)

    # Process frames
    print("Applying stabilization...")
    for i, frame in enumerate(frames):
        stabilized_frame = stabilizer.stabilize_frame(frame, i)
        out.write(stabilized_frame)

        # Display progress
        if i % 20 == 0:  # Update progress every 20 frames
            progress = (i + 1) / len(frames) * 100
            print(f"Progress: {progress:.1f}%")

        # Display frames
        display_width = 1200
        aspect_ratio = frame.shape[1] / frame.shape[0]
        display_height = int(display_width / aspect_ratio / 2)
        
        frame_resized = cv2.resize(frame, (display_width//2, display_height))
        stabilized_resized = cv2.resize(stabilized_frame, (display_width//2, display_height))
        
        combined_frame = np.hstack((frame_resized, stabilized_resized))
        cv2.putText(combined_frame, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, 'Stabilized', (display_width//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Video Stabilization', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done! Stabilized video saved as:", output_path)

if __name__ == "__main__":
    main()
