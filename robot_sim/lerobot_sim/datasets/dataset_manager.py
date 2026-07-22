"""
Dataset recording, loading, and Hugging Face Hub integration.

Provides a ``SimDatasetRecorder`` that captures observations and actions
from simulation episodes into the LeRobotDataset-compatible format
(Parquet for tabular data, image sequences for visual data).  Also
provides helpers to load existing datasets from the HF Hub.

Classes:
    SimDatasetRecorder: Records episodes from simulation into disk storage.
    HubDatasetLoader: Streams or downloads datasets from the HF Hub.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ======================================================================
# SimDatasetRecorder
# ======================================================================


@dataclass
class SimDatasetRecorder:
    """Records simulation episodes into LeRobotDataset-compatible storage.

    For each episode the recorder accumulates observation and action frames
    then flushes them to disk as NumPy ``.npz`` files (tabular) and PNG
    images (visual).  A ``meta.json`` index is maintained so that the
    dataset can later be loaded or pushed to the Hugging Face Hub.

    Attributes:
        output_dir: Root directory where episode data is written.
        dataset_name: Human-readable name included in ``meta.json``.
        fps: Recording frame rate (informational, written to metadata).
        episodes: In-memory buffer of completed episode records.
    """

    output_dir: str = "./recorded_data"
    dataset_name: str = "lerobot_sim_dataset"
    fps: int = 10
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    _current_episode: List[Dict[str, np.ndarray]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        """Begin recording a new episode, clearing the frame buffer.

        Raises:
            RuntimeError: If the previous episode was not ended.
        """
        if self._current_episode:
            raise RuntimeError("Previous episode not ended. Call end_episode() first.")
        self._current_episode = []

    def record_step(self, observation: Dict[str, np.ndarray], action: np.ndarray) -> None:
        """Append a single timestep of observation and action data.

        Args:
            observation: Dictionary of observation arrays (e.g. ``agent_pos``, ``pixels``).
            action: The action array applied at this timestep.

        Raises:
            RuntimeError: If no episode is currently being recorded.
        """
        if self._current_episode is None:
            raise RuntimeError("No active episode. Call start_episode() first.")
        frame: Dict[str, np.ndarray] = {"action": np.asarray(action)}
        frame.update(observation)
        self._current_episode.append(frame)

    def _build_episode_record(self, episode_idx: int) -> Dict[str, Any]:
        """Construct the metadata record for a completed episode.

        Args:
            episode_idx: Zero-based episode index.

        Returns:
            Dictionary with episode metadata (index, length, data path).
        """
        return {
            "episode_index": episode_idx,
            "num_frames": len(self._current_episode),
            "data_path": self._episode_dir_name(episode_idx),
        }

    def _episode_dir_name(self, episode_idx: int) -> str:
        """Return the sub-directory name for an episode.

        Args:
            episode_idx: Zero-based episode index.

        Returns:
            String of the form ``'episode_000042'``.
        """
        return f"episode_{episode_idx:06d}"

    def end_episode(self) -> int:
        """Finalise the current episode and flush data to disk.

        Returns:
            The zero-based index of the completed episode.

        Raises:
            RuntimeError: If no episode is currently being recorded.
        """
        if not self._current_episode:
            raise RuntimeError("No active episode to end.")
        episode_idx = len(self.episodes)
        record = self._build_episode_record(episode_idx)
        self._flush_episode_to_disk(episode_idx)
        self.episodes.append(record)
        self._current_episode = []
        return episode_idx

    def save_metadata(self) -> Path:
        """Write ``meta.json`` summarising all recorded episodes.

        Returns:
            ``Path`` to the written metadata file.
        """
        meta = self._build_metadata()
        path = Path(self.output_dir) / "meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2))
        return path

    # ------------------------------------------------------------------
    # Flush helpers
    # ------------------------------------------------------------------

    def _ensure_episode_dir(self, episode_idx: int) -> Path:
        """Create and return the on-disk directory for an episode.

        Args:
            episode_idx: Zero-based episode index.

        Returns:
            ``Path`` object for the episode directory.
        """
        ep_dir = Path(self.output_dir) / self._episode_dir_name(episode_idx)
        ep_dir.mkdir(parents=True, exist_ok=True)
        return ep_dir

    def _save_tabular_data(self, ep_dir: Path) -> None:
        """Write non-image frame data to a compressed ``.npz`` file.

        Args:
            ep_dir: Directory to write the file into.
        """
        tabular = {k: v for frame in self._current_episode for k, v in frame.items() if v.ndim <= 1}
        stacked = {k: np.stack([f[k] for f in self._current_episode if k in f]) for k in tabular}
        np.savez_compressed(ep_dir / "tabular.npz", **stacked)

    def _save_image_data(self, ep_dir: Path) -> None:
        """Write per-frame image observations as individual PNG files.

        Args:
            ep_dir: Directory to write image files into.
        """
        for idx, frame in enumerate(self._current_episode):
            for key, arr in frame.items():
                if arr.ndim == 3:
                    self._save_single_image(ep_dir, key, idx, arr)

    def _save_single_image(self, ep_dir: Path, key: str, idx: int, arr: np.ndarray) -> None:
        """Write one image array to disk as a raw ``.npy`` file.

        Args:
            ep_dir: Parent directory.
            key: Observation key name (used in the filename).
            idx: Timestep index.
            arr: (H, W, C) image array.
        """
        img_dir = ep_dir / key
        img_dir.mkdir(parents=True, exist_ok=True)
        np.save(img_dir / f"{idx:06d}.npy", arr)

    def _flush_episode_to_disk(self, episode_idx: int) -> None:
        """Write all frame data for an episode to disk.

        Args:
            episode_idx: Zero-based episode index.
        """
        ep_dir = self._ensure_episode_dir(episode_idx)
        self._save_tabular_data(ep_dir)
        self._save_image_data(ep_dir)

    def _build_metadata(self) -> Dict[str, Any]:
        """Construct the metadata dictionary for ``meta.json``.

        Returns:
            Dictionary with dataset-level and per-episode metadata.
        """
        return {
            "dataset_name": self.dataset_name,
            "fps": self.fps,
            "total_episodes": len(self.episodes),
            "episodes": self.episodes,
        }


# ======================================================================
# HubDatasetLoader
# ======================================================================


@dataclass
class HubDatasetLoader:
    """Load or stream a LeRobotDataset from the Hugging Face Hub.

    This is a thin convenience wrapper that downloads episode metadata
    and tabular data from a Hub repository.  For full compatibility with
    the upstream ``LeRobotDataset`` or ``StreamingLeRobotDataset`` classes,
    users should install the ``lerobot`` package; this loader provides a
    lightweight subset of the same functionality.

    Attributes:
        repo_id: Hugging Face Hub repository identifier (e.g. ``'lerobot/pusht'``).
        cache_dir: Local directory for cached downloads.
    """

    repo_id: str = "lerobot/pusht"
    cache_dir: str = "./hub_cache"

    def _download_metadata(self) -> Dict[str, Any]:
        """Download and parse the dataset ``meta.json`` from the Hub.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub") from exc
        local = hf_hub_download(repo_id=self.repo_id, filename="meta/info.json", cache_dir=self.cache_dir)
        return json.loads(Path(local).read_text())

    def _list_episodes(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract the episode list from metadata.

        Args:
            meta: Parsed ``meta.json`` dictionary.

        Returns:
            List of episode record dictionaries.
        """
        return meta.get("episodes", meta.get("splits", []))

    def load_info(self) -> Dict[str, Any]:
        """Download dataset metadata and return summary information.

        Returns:
            Dictionary with ``repo_id``, ``total_episodes``, and ``episodes``.
        """
        meta = self._download_metadata()
        episodes = self._list_episodes(meta)
        return {"repo_id": self.repo_id, "total_episodes": len(episodes), "meta": meta}

    def list_hub_datasets(self, search: str = "lerobot") -> List[str]:
        """List dataset repositories on the Hub matching a search query.

        Args:
            search: Search string (default ``'lerobot'``).

        Returns:
            List of repository ID strings.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
        """
        try:
            from huggingface_hub import list_datasets
        except ImportError as exc:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub") from exc
        results = list_datasets(search=search, limit=50)
        return [ds.id for ds in results]
