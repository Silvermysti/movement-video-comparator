import cv2
import mediapipe as mp
import os
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import subprocess

class DanceVideoComparator:
    def __init__(self, min_detection_confidence=0.5):
        """Initialize dance video comparator"""
        # Colors for each video
        self.video1_color = (0, 255, 0)    # Green
        self.video2_color = (0, 165, 255)  # Orange
        
        # Check for model file
        self.model_path = 'pose_landmarker.task'
        if not os.path.exists(self.model_path):
            print(f"Error: Model file '{self.model_path}' not found!")
            raise FileNotFoundError(f"Model file '{self.model_path}' not found")
        
        self.min_detection_confidence = min_detection_confidence
        self.mirror_video2 = False
        
        # Store processed video info
        self.processed_video1_path = None
        self.processed_video2_path = None
        
    # STEP 1: Import videos (already done via file paths)
    
    def standardize_fps(self, video_path, target_fps=30):
        """STEP 2: Bring videos to same frame rate (30 fps)"""
        print(f"Standardizing FPS for {os.path.basename(video_path)}...")
        
        # Get original FPS
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if original_fps == target_fps:
            print(f"  Already at {target_fps} FPS, no conversion needed")
            return video_path
        
        # Create temporary video file with standard FPS
        with tempfile.NamedTemporaryFile(suffix='_30fps.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        try:
            # Use ffmpeg to convert FPS
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-filter:v', f'fps={target_fps}',
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'fast',
                '-y',
                temp_video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  FPS conversion failed: {result.stderr}")
                return video_path
            
            print(f"  Converted from {original_fps:.1f} FPS to {target_fps} FPS")
            return temp_video_path
            
        except Exception as e:
            print(f"  Error converting FPS: {e}")
            return video_path
    
    # STEP 3: Ask about mirroring
    def ask_mirror_option(self):
        """Ask user if video 2 should be mirrored"""
        print("\n=== Mirror Option ===")
        print("Mirror Video 2 if dancers face opposite directions.")
        print("Video will be mirrored BEFORE pose detection.")
        
        while True:
            response = input("Mirror Video 2? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                self.mirror_video2 = True
                print("Video 2 will be mirrored.")
                break
            elif response in ['n', 'no']:
                self.mirror_video2 = False
                print("Video 2 will not be mirrored.")
                break
            else:
                print("Please enter 'y' or 'n'.")
    
    def create_landmarker(self, running_mode=vision.RunningMode.VIDEO):
        """Create a pose landmarker instance"""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            min_pose_detection_confidence=self.min_detection_confidence
        )
        return vision.PoseLandmarker.create_from_options(options)
    
    def detect_pose_in_frame(self, frame, landmarker, mirror=False):
        """Detect pose in a single frame"""
        # Apply mirroring BEFORE pose detection if needed
        if mirror:
            frame = cv2.flip(frame, 1)  # Horizontal flip
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = landmarker.detect(mp_image)
        
        if detection_result.pose_landmarks:
            return detection_result.pose_landmarks[0]
        return None
    
    def calculate_angle(self, point_a, point_b, point_c):
        """Calculate the angle at point B formed by points A-B-C"""
        # Convert to numpy arrays
        a = np.array([point_a.x, point_a.y])
        b = np.array([point_b.x, point_b.y])
        c = np.array([point_c.x, point_c.y])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Clamp to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Return angle in degrees
        return np.degrees(np.arccos(cosine_angle))
    
    def extract_all_angles(self, landmarks):
        """Extract ALL joint angles from pose landmarks"""
        if landmarks is None:
            return None
        
        angles = []
        
        # Define ALL possible angle triplets from pose landmarks (0-32)
        # We'll create angles for all meaningful joint combinations
        angle_definitions = []
        
        # Major joints and their connections
        joints = {
            'left_shoulder': 11, 'left_elbow': 13, 'left_wrist': 15,
            'right_shoulder': 12, 'right_elbow': 14, 'right_wrist': 16,
            'left_hip': 23, 'left_knee': 25, 'left_ankle': 27,
            'right_hip': 24, 'right_knee': 26, 'right_ankle': 28,
            'nose': 0
        }
        
        # Create angle definitions for limbs
        # Left arm
        angle_definitions.extend([
            (joints['left_shoulder'], joints['left_elbow'], joints['left_wrist']),  # Elbow angle
            (joints['left_hip'], joints['left_shoulder'], joints['left_elbow']),    # Shoulder angle
        ])
        
        # Right arm
        angle_definitions.extend([
            (joints['right_shoulder'], joints['right_elbow'], joints['right_wrist']),  # Elbow angle
            (joints['right_hip'], joints['right_shoulder'], joints['right_elbow']),    # Shoulder angle
        ])
        
        # Left leg
        angle_definitions.extend([
            (joints['left_hip'], joints['left_knee'], joints['left_ankle']),  # Knee angle
            (joints['left_shoulder'], joints['left_hip'], joints['left_knee']),  # Hip angle
        ])
        
        # Right leg
        angle_definitions.extend([
            (joints['right_hip'], joints['right_knee'], joints['right_ankle']),  # Knee angle
            (joints['right_shoulder'], joints['right_hip'], joints['right_knee']),  # Hip angle
        ])
        
        # Body angles
        angle_definitions.extend([
            (joints['left_shoulder'], joints['left_hip'], joints['right_hip']),  # Torso angle
            (joints['right_shoulder'], joints['right_hip'], joints['left_hip']),  # Torso angle
            (joints['left_elbow'], joints['left_shoulder'], joints['right_shoulder']),  # Chest angle
            (joints['right_elbow'], joints['right_shoulder'], joints['left_shoulder']),  # Chest angle
        ])
        
        # Calculate each angle
        for a_idx, b_idx, c_idx in angle_definitions:
            if (a_idx < len(landmarks) and b_idx < len(landmarks) and c_idx < len(landmarks)):
                angle = self.calculate_angle(landmarks[a_idx], landmarks[b_idx], landmarks[c_idx])
                angles.append(angle)
            else:
                angles.append(0.0)  # Default if landmark missing
        
        return np.array(angles)
    
    def create_angle_arrays(self, video_path, frame_gap=10, mirror=False):
        """STEP 4: Create array of angles at every frame_gap interval"""
        print(f"\nCreating angle array for {os.path.basename(video_path)}...")
        print(f"  Sampling every {frame_gap} frames")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create landmarker for pose detection
        landmarker = self.create_landmarker(vision.RunningMode.IMAGE)
        
        angle_array = []
        sampled_frame_numbers = []
        frame_count = 0
        samples_count = 0
        
        print(f"  Total frames: {total_frames}")
        print(f"  Expected samples: ~{total_frames // frame_gap}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every frame_gap frames (e.g., every 10th frame)
            if frame_count % frame_gap == 0:
                # Detect pose in frame
                landmarks = self.detect_pose_in_frame(frame, landmarker, mirror)
                
                # Extract ALL angles
                angles = self.extract_all_angles(landmarks)
                
                if angles is not None:
                    angle_array.append(angles)
                    sampled_frame_numbers.append(frame_count)
                    samples_count += 1
                
                # Show progress
                if samples_count % 20 == 0:
                    print(f"    Processed {samples_count} samples...")
            
            frame_count += 1
        
        cap.release()
        landmarker.close()
        
        # Convert to numpy array
        angle_array_np = np.array(angle_array)
        print(f"  Completed: {angle_array_np.shape[0]} samples, {angle_array_np.shape[1]} angles per sample")
        print(f"  Frame numbers sampled: {sampled_frame_numbers[:5]}... (first 5)")
        
        return angle_array_np, sampled_frame_numbers
    
    def find_best_match(self, angles1, angles2):
        """STEP 5: Find best matching sections between two angle arrays"""
        print("\n=== Finding best alignment ===")
        
        len1 = len(angles1)
        len2 = len(angles2)
        
        # Determine comparison length (shorter array length)
        max_comparison_len = min(len1, len2)  # Compare up to 50 samples
        print(f"  Array 1 length: {len1} samples")
        print(f"  Array 2 length: {len2} samples")
        print(f"  Comparing sections up to {max_comparison_len} samples long")
        
        best_score = -1
        best_offset = 0
        best_length = 0
        best_start1 = 0
        best_start2 = 0
        
        # Try different section lengths
        for section_len in range(int(max_comparison_len*3/4), max_comparison_len + 1, 2):
            print(f"  Testing section length: {section_len} samples")
            
            # Slide window through array1
            for start1 in range(0, len1 - section_len + 1, 2):  # Step by 2 for speed
                section1 = angles1[start1:start1 + section_len]
                
                # Slide window through array2
                for start2 in range(0, len2 - section_len + 1, 2):  # Step by 2 for speed
                    section2 = angles2[start2:start2 + section_len]
                    
                    # Calculate similarity score
                    score = self.calculate_section_similarity(section1, section2)
                    
                    if score > best_score:
                        best_score = score
                        best_start1 = start1
                        best_start2 = start2
                        best_length = section_len
        
        # Calculate offset in FRAMES (not samples)
        # best_start1 and best_start2 are SAMPLE indices
        # Each sample represents 10 frames (frame_gap)
        print(f"\nBest match found:")
        print(f"  Score: {best_score:.3f}")
        print(f"  Section length: {best_length} samples ({best_length * 10} frames)")
        print(f"  Array1 start at sample: {best_start1} (frame {best_start1 * 10})")
        print(f"  Array2 start at sample: {best_start2} (frame {best_start2 * 10})")
        
        # Calculate frame offset
        # If Array2 starts at sample 5 and Array1 at sample 1
        # Then Array2 is (5-1)*10 = 40 frames ahead of Array1
        # Or Array1 needs to start 40 frames later than Array2

        print(f"\nDEBUG - First sample angles:")
        print(f"Video1 angles[0]: {angles1[0][:4]}")  # First 4 angles
        print(f"Video2 angles[0]: {angles2[0][:4]}")  # First 4 angles
        print(f"Similarity: {self.calculate_section_similarity([angles1[0]], [angles2[0]])}")
        
        frame_offset = (best_start2 - best_start1) * 10
        
        print(f"  Frame offset: {frame_offset} frames")
        if frame_offset >= 0:
            print(f"  Interpretation: Video 2 starts {frame_offset} frames AFTER Video 1")
        else:
            print(f"  Interpretation: Video 1 starts {abs(frame_offset)} frames AFTER Video 2")
        
        return frame_offset
    
    def calculate_section_similarity(self, section1, section2):
        """Calculate similarity between two angle sections"""
        # Simple cosine similarity averaged over all frames in section
        similarities = []
        
        for i in range(len(section1)):
            vec1 = section1[i]
            vec2 = section2[i]
            
            # Skip if vectors are all zeros
            if np.linalg.norm(vec1) < 1e-6 or np.linalg.norm(vec2) < 1e-6:
                continue
            
            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(similarity)
        
        if len(similarities) == 0:
            return 0
        
        # Return average similarity
        return np.mean(similarities)
    
    def draw_landmarks(self, frame, landmarks, color):
        """Draw pose landmarks on frame"""
        POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), 
            (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), 
            (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), 
            (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), 
            (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), 
            (29, 31), (30, 32), (27, 31), (28, 32)
        ]
        
        if landmarks is None:
            return frame
        
        # Draw landmarks as circles
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, color, 2)
        
        # Draw connections as lines
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_x = int(landmarks[start_idx].x * frame.shape[1])
                start_y = int(landmarks[start_idx].y * frame.shape[0])
                end_x = int(landmarks[end_idx].x * frame.shape[1])
                end_y = int(landmarks[end_idx].y * frame.shape[0])
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
        
        return frame
    
    def save_videos_aligned(self, video1_path, video2_path, frame_offset, output_path="comparison_output.mp4", frame_gap=10):
        """STEP 6: Save both videos side by side according to offset as MP4 file"""
        print("\n=== Saving aligned videos to MP4 ===")
        print(f"Frame offset: {frame_offset} frames")
        print(f"Output file: {output_path}")
        
        # Open videos
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        # Get video properties
        fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
        fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
        width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video 1: {width1}x{height1}, {fps1} FPS")
        print(f"Video 2: {width2}x{height2}, {fps2} FPS")
        # Set starting frames based on offset
        if frame_offset >= 0:
            # Video2 starts later
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
            start_frame = 0
            print(f"Video 2 starts {frame_offset} frames after Video 1")
        else:
            # Video1 starts later
            cap1.set(cv2.CAP_PROP_POS_FRAMES, -frame_offset)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            start_frame = -frame_offset
            print(f"Video 1 starts {abs(frame_offset)} frames after Video 2")
        
        # Calculate display size
        max_display_width = 1280
        max_display_height = 720
        total_width_needed = width1 + width2 + 40
        
        if total_width_needed > max_display_width:
            scale_factor = max_display_width / total_width_needed
            target_height = int(max(height1, height2) * scale_factor)
        else:
            target_height = max(height1, height2)
        
        target_width1 = int(width1 * (target_height / height1))
        target_width2 = int(width2 * (target_height / height2))
        
        combined_width = target_width1 + target_width2 + 40
        combined_height = target_height + 80
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = 30  # Use 30 FPS for output
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (combined_width, combined_height))
        
        print(f"\nOutput video: {combined_width}x{combined_height}, {output_fps} FPS")
        print("Processing frames...")
        
        # Create landmarkers for pose detection
        landmarker1 = self.create_landmarker(vision.RunningMode.VIDEO)
        landmarker2 = self.create_landmarker(vision.RunningMode.VIDEO)
        
        frame_count = start_frame
        total_frames_to_process = min(
            int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) - (0 if frame_offset >= 0 else -frame_offset),
            int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) - (frame_offset if frame_offset >= 0 else 0)
        )
        
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Apply mirroring to frame2 BEFORE pose detection if needed
            if self.mirror_video2:
                frame2 = cv2.flip(frame2, 1)
            
            # Resize frames
            frame1_resized = cv2.resize(frame1, (target_width1, target_height))
            frame2_resized = cv2.resize(frame2, (target_width2, target_height))
            
            # Detect poses
            timestamp = (frame_count - start_frame) * 1000
            landmarks1 = self.detect_pose_for_video(frame1_resized, landmarker1, timestamp)
            landmarks2 = self.detect_pose_for_video(frame2_resized, landmarker2, timestamp)
            
            # Draw poses
            if landmarks1:
                frame1_resized = self.draw_landmarks(frame1_resized, landmarks1, self.video1_color)
            if landmarks2:
                frame2_resized = self.draw_landmarks(frame2_resized, landmarks2, self.video2_color)
            
            # Create combined frame
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            gap = 20
            combined_frame[:target_height, :target_width1] = frame1_resized
            combined_frame[:target_height, target_width1 + gap:target_width1 + gap + target_width2] = frame2_resized
            
            # Add separator line
            separator_x = target_width1 + gap // 2
            cv2.line(combined_frame, (separator_x, 0), 
                    (separator_x, target_height), (255, 255, 255), 2)
            
            # Add labels
            label_y = 30
            cv2.putText(combined_frame, "15", 
                    (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.video1_color, 2)
            
            video2_label = "DANCE" + (" (MIRRORED)" if self.mirror_video2 else "")
            cv2.putText(combined_frame, video2_label, 
                    (target_width1 + gap + 10, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.video2_color, 2)
            
            # Add info panel
            info_y = target_height + 20
            line_height = 20
            
            info_lines = [
                f"Frame: {frame_count - start_frame}",
                f"Offset: {frame_offset} frames",
                f"Playback: {output_fps} FPS",
                f"Mirror: {'ON' if self.mirror_video2 else 'OFF'}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(combined_frame, line, 
                        (10, info_y + i * line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame to output video
            out.write(combined_frame)
            
            # Show progress
            if (frame_count - start_frame) % 30 == 0:  # Every second
                progress = (frame_count - start_frame) / total_frames_to_process * 100
                print(f"  Progress: {progress:.1f}% ({frame_count - start_frame}/{total_frames_to_process} frames)")
            
            frame_count += 1
        
        # Cleanup
        cap1.release()
        cap2.release()
        out.release()
        landmarker1.close()
        landmarker2.close()
        
        print(f"\nVideo saved successfully: {output_path}")
        print(f"Total frames processed: {frame_count - start_frame}")
        
        # Convert to better compression using ffmpeg if available
        try:
            print("\nOptimizing video file...")
            optimized_path = output_path.replace('.mp4', '_optimized.mp4')
            cmd = [
                'ffmpeg',
                '-i', output_path,
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '23',
                '-c:a', 'copy',
                '-y',
                optimized_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                os.remove(output_path)
                os.rename(optimized_path, output_path)
                print("Video optimized for better compression.")
            else:
                print("Original video saved (optimization skipped).")
        except:
            print("Original video saved (ffmpeg not available for optimization).")
    
    def detect_pose_for_video(self, frame, landmarker, timestamp_ms):
        """Detect pose for video processing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if detection_result.pose_landmarks:
            return detection_result.pose_landmarks[0]
        return None
    
    def cleanup_temp_files(self):
        """Clean up temporary video files"""
        if self.processed_video1_path and os.path.exists(self.processed_video1_path):
            try:
                os.unlink(self.processed_video1_path)
                print(f"Cleaned up temp file: {os.path.basename(self.processed_video1_path)}")
            except:
                pass
        
        if self.processed_video2_path and os.path.exists(self.processed_video2_path):
            try:
                os.unlink(self.processed_video2_path)
                print(f"Cleaned up temp file: {os.path.basename(self.processed_video2_path)}")
            except:
                pass
    
    def compare_videos(self, video1_path, video2_path):
        """Main comparison pipeline"""
        print(f"\nVideo 1: {os.path.basename(video1_path)}")
        print(f"Video 2: {os.path.basename(video2_path)}")
        
        try:
            # STEP 2: Bring both to same frame rate (30 fps)
            print("\n=== STEP 2: Standardizing FPS ===")
            processed_video1_path = self.standardize_fps(video1_path, target_fps=30)
            processed_video2_path = self.standardize_fps(video2_path, target_fps=30)
            
            # Store paths for cleanup
            self.processed_video1_path = processed_video1_path if processed_video1_path != video1_path else None
            self.processed_video2_path = processed_video2_path if processed_video2_path != video2_path else None
            
            # STEP 3: Ask about mirroring
            self.ask_mirror_option()
            
            # STEP 4: Create angle arrays (every 10 frames)
            print("\n=== STEP 4: Creating angle arrays ===")
            print("Sampling every 10 frames...")
            
            angles1, _ = self.create_angle_arrays(processed_video1_path, frame_gap=10, mirror=False)
            angles2, _ = self.create_angle_arrays(processed_video2_path, frame_gap=10, mirror=self.mirror_video2)
            
            # STEP 5: Find best match and calculate offset
            print("\n=== STEP 5: Finding best alignment ===")
            frame_offset = self.find_best_match(angles1, angles2)
            
            # STEP 6: Save videos aligned to MP4 file
            print("\n=== STEP 6: Saving aligned video to file ===")
            output_filename = f"(old)comparison_{os.path.splitext(os.path.basename(video1_path))[0]}_vs_{os.path.splitext(os.path.basename(video2_path))[0]}.mp4"
            self.save_videos_aligned(processed_video1_path, processed_video2_path, frame_offset, output_filename)
            
        finally:
            # Cleanup
            self.cleanup_temp_files()

def main():
    """Main function"""
    print("=== Dance Video Comparison Tool ===")
    print("Compare two dance videos using pose angles")
    print("Output will be saved as an MP4 file")
    
    # List video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir('.') if f.lower().endswith(video_extensions)]
    
    if len(video_files) < 2:
        print("Need at least 2 video files!")
        return
    
    print("\nAvailable videos:")
    for i, video in enumerate(video_files, 1):
        print(f"{i}. {video}")
    
    # Select videos
    try:
        choice1 = int(input("\nSelect reference video (number): ")) - 1
        if choice1 < 0 or choice1 >= len(video_files):
            print("Invalid selection")
            return
        
        # Show remaining videos
        remaining = [v for i, v in enumerate(video_files) if i != choice1]
        print("\nRemaining videos:")
        for i, video in enumerate(remaining, 1):
            print(f"{i}. {video}")
        
        choice2 = int(input("Select dance video to compare (number): ")) - 1
        if choice2 < 0 or choice2 >= len(remaining):
            print("Invalid selection")
            return
        
        video1_path = video_files[choice1]
        video2_path = remaining[choice2]
        
    except ValueError:
        print("Please enter numbers")
        return
    
    # Run comparator
    try:
        comparator = DanceVideoComparator(min_detection_confidence=0.5)
        comparator.compare_videos(video1_path, video2_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nProgram finished.")

if __name__ == "__main__":
    main()