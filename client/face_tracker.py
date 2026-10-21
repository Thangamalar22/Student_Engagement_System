import time
import math
import numpy as np

class TrackedFace:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        self.last_seen = time.time()
        self.missed_frames = 0
        self.label = f"Student {self.track_id:02d}"

class FaceTracker:
    def __init__(self, max_distance=150.0, max_missed=15):
        """
        max_distance: max pixel distance to consider a match
        max_missed: how many consecutive frames a face can be missing before track is dropped
        """
        self.tracks = []
        self.next_id = 1
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, detections, timestamp=None):
        """
        detections: list of dicts. Each dict MUST have:
                    'bbox': (x, y, w, h) in pixels
                    It can also have other data like 'landmarks'.
        Returns: list of dicts, original detections updated with 'track_id' and 'label'
        """
        if timestamp is None:
            timestamp = time.time()

        if not detections:
            # If no detections, increment missed frames on all tracks
            active_tracks = []
            for track in self.tracks:
                track.missed_frames += 1
                if track.missed_frames <= self.max_missed:
                    active_tracks.append(track)
            self.tracks = active_tracks
            return []
            
        current_centers = [(d['bbox'][0] + d['bbox'][2]/2.0, d['bbox'][1] + d['bbox'][3]/2.0) for d in detections]
        
        matched_indices = set()
        active_tracks = []
        
        # Match existing tracks to detections
        for track in self.tracks:
            best_idx = -1
            best_dist = float('inf')
            
            for i, center in enumerate(current_centers):
                if i in matched_indices:
                    continue
                dist = math.hypot(center[0] - track.center[0], center[1] - track.center[1])
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_idx = i
                    
            if best_idx != -1:
                matched_indices.add(best_idx)
                d = detections[best_idx]
                track.bbox = d['bbox']
                track.center = current_centers[best_idx]
                track.last_seen = timestamp
                track.missed_frames = 0
                
                d['track_id'] = track.track_id
                d['label'] = track.label
                active_tracks.append(track)
            else:
                track.missed_frames += 1
                if track.missed_frames <= self.max_missed:
                    active_tracks.append(track)
                    
        # Create new tracks for unmatched detections
        for i, d in enumerate(detections):
            if i not in matched_indices:
                new_track = TrackedFace(self.next_id, d['bbox'])
                self.next_id += 1
                active_tracks.append(new_track)
                d['track_id'] = new_track.track_id
                d['label'] = new_track.label
                
        self.tracks = active_tracks
        return detections

if __name__ == "__main__":
    tracker = FaceTracker(max_distance=50, max_missed=2)
    
    # Frame 1: One face
    print("Frame 1: One face at (10,10)")
    dets1 = [{'bbox': (10, 10, 50, 50)}]
    tracked1 = tracker.update(dets1)
    for t in tracked1:
        print(f"  {t['label']} at {t['bbox']}")
        
    # Frame 2: Same face moved slightly, plus a new face
    print("\nFrame 2: Face 1 moved to (12,12), new face at (100,100)")
    dets2 = [{'bbox': (12, 12, 50, 50)}, {'bbox': (100, 100, 40, 40)}]
    tracked2 = tracker.update(dets2)
    for t in tracked2:
        print(f"  {t['label']} at {t['bbox']}")
        
    # Frame 3: First face lost, second face moved slightly
    print("\nFrame 3: Face 1 lost, Face 2 moved to (105,105)")
    dets3 = [{'bbox': (105, 105, 40, 40)}]
    tracked3 = tracker.update(dets3)
    for t in tracked3:
        print(f"  {t['label']} at {t['bbox']}")
        
    # Frame 4: Second face lost completely (no detections)
    print("\nFrame 4: No detections")
    tracked4 = tracker.update([])
    print(f"  Active tracks returned: {len(tracked4)}")
    print(f"  Active tracks in memory: {len(tracker.tracks)}")
