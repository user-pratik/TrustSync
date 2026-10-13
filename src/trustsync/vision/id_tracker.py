"""Multi-face track association using Hungarian algorithm."""
from __future__ import annotations

from typing import Any
import numpy as np
import scipy.optimize


class IdTracker:
    """Greedy Hungarian tracker matching bounding-box centroids across frames."""

    def __init__(self, max_dist: float, max_missed: int) -> None:
        self.max_dist = max_dist
        self.max_missed = max_missed
        self._next_id = 0
        self._tracks: dict[int, dict[str, Any]] = {}

    def assign(self, centroids: np.ndarray) -> list[int]:
        """Assign persistent track IDs to detections.

        Args:
            centroids: np.ndarray shape (n, 2) normalized face centroids.

        Returns:
            list[int]: Track IDs assigned to each detection in order.
        """
        n = len(centroids)
        assigned: list[int | None] = [None] * n

        if not self._tracks:
            # No existing tracks: all detections get new sequential IDs
            for i in range(n):
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "centroid": centroids[i].copy(),
                    "missed": 0,
                }
                assigned[i] = tid
            return [tid for tid in assigned if tid is not None]

        track_ids = list(self._tracks.keys())
        m = len(track_ids)

        if n > 0:
            cost_matrix = np.zeros((m, n), dtype=np.float64)
            for r, tid in enumerate(track_ids):
                t_pos = self._tracks[tid]["centroid"]
                cost_matrix[r, :] = np.linalg.norm(centroids - t_pos, axis=1)

            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            matched_tracks: set[int] = set()

            for r, c in zip(row_ind, col_ind):
                dist = cost_matrix[r, c]
                if dist <= self.max_dist:
                    tid = track_ids[r]
                    self._tracks[tid]["centroid"] = centroids[c].copy()
                    self._tracks[tid]["missed"] = 0
                    assigned[c] = tid
                    matched_tracks.add(tid)

            # Unmatched detections get new IDs
            for c in range(n):
                if assigned[c] is None:
                    tid = self._next_id
                    self._next_id += 1
                    self._tracks[tid] = {
                        "centroid": centroids[c].copy(),
                        "missed": 0,
                    }
                    assigned[c] = tid

            # Increment missed count for tracks not matched
            for tid in track_ids:
                if tid not in matched_tracks:
                    self._tracks[tid]["missed"] += 1
        else:
            # No detections: all tracks increment missed counter
            for tid in track_ids:
                self._tracks[tid]["missed"] += 1

        # Delete tracks missed for more than max_missed consecutive calls
        dead_tracks = [
            tid
            for tid, t_data in self._tracks.items()
            if t_data["missed"] > self.max_missed
        ]
        for tid in dead_tracks:
            del self._tracks[tid]

        return [tid for tid in assigned if tid is not None]
