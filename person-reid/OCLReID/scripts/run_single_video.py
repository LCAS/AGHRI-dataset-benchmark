import os
import os.path as osp
import numpy as np
import json as js
import tempfile
import hashlib
from argparse import ArgumentParser
import sys
from pathlib import Path

file_path = Path(__file__).resolve()
PROJECT_ROOT = file_path.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mmcv
from mmtrack.apis import inference_mot, init_model, inference_sot, extract_feature
from mmtrack.core import imshow_tracks, results2outs
from mmtrack.utils import get_root_logger
from tqdm import main
from multiprocessing import Pool
import threading
import time
from multiprocessing import Process, Manager
import torch
import math
import cv2
import shutil

from utils.visdom import Visdom
from utils import Config
from utils.visualization import Drawer
from scripts.evaluator import Tracker

import fitlog
import random
import torchvision.transforms as T
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.contrib import tzip
import sklearn
import matplotlib.cm as cm
from mmtrack.models.reid.utils import GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS
import json
from tqdm import tqdm
DEFAULT_REID_CHECKPOINT = PROJECT_ROOT / "checkpoints/reid/resnet18.pth"


def unwrap_checkpoint_state(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj if isinstance(obj, dict) else {}


def hash_tensors(tensors):
    digest = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_tensor_hash(checkpoint_path):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = unwrap_checkpoint_state(checkpoint)
    tensors = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    return hash_tensors(tensors), tensors

def write_to_json(file_path, data):
    """
    Writes the tracking results dictionary to a JSON file.
    Overwrites the file with the most recent data.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def graphical_display_available():
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    return True


def diagnostic_jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): diagnostic_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def diagnostic_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def diagnostic_cosine(a, b):
    if a is None or b is None:
        return None
    try:
        a = a.detach().flatten().float().cpu()
        b = b.detach().flatten().float().cpu()
        denom = float(torch.norm(a).item() * torch.norm(b).item())
        if denom <= 0:
            return None
        return float(torch.dot(a, b).item() / denom)
    except Exception:
        return None

class TargetIdentificationEvaluator():
    def __init__(self, hyper_config, config, identifier_config):
        self.type = "rpf"
        self.ckpt = None
        self.config = config
        self.hyper_config = hyper_config
        self.identifier_config = identifier_config
        self.identifier_params = Config.fromfile(identifier_config)
        self.tracker = None

        self.input = hyper_config.input
        self.output = hyper_config.output
        self.output_json = hyper_config.output_json
        self.visualization_video = getattr(hyper_config, 'visualization_video', None)
        self.gt_bbox_file = hyper_config.gt_bbox_file
        self.seed = hyper_config.seed
        self.show_result = hyper_config.show_result
        self.start_frame = getattr(hyper_config, 'start_frame', 0)
        self.method = getattr(hyper_config, 'method', 'OCLReID')
        self.live_display_enabled = bool(self.show_result)
        self.live_display_failed = False
        self.live_paused = False
        self.live_window_name = "OCLReID Live"
        self.visualization_writer = None
        self.visualization_writer_enabled = bool(self.visualization_video)
        self.visualization_fps = 30.0
        self.reid_checkpoint = Path(
            getattr(hyper_config, 'reid_checkpoint', DEFAULT_REID_CHECKPOINT)
        )
        self.reid_loaded_keys = []
        self.reid_checkpoint_hash = None
        self.reid_initial_loaded_hash = None
        self.reid_initial_full_hash = None
        diagnostic_log = getattr(hyper_config, 'diagnostic_log', None)
        self.diagnostic_log = Path(diagnostic_log) if diagnostic_log else None
        self.save_full_diagnostic_features = bool(
            getattr(hyper_config, 'save_full_diagnostic_features', False)
        )
        self._last_diagnostic_memory_sizes = None

        self.fps = 1000
        self.result = {}

    def get_identifier(self):
        if self.tracker is None:
            return None
        obj_tracker = getattr(self.tracker, "obj_tracker", None)
        return getattr(obj_tracker, "identifier", None)

    def get_diagnostic_memory_sizes(self, identifier):
        sizes = {}
        classifier = getattr(identifier, "classifier", None)
        memory_manager = getattr(classifier, "memory_manager", None)
        memories = getattr(memory_manager, "memory", {}) if memory_manager is not None else {}
        for name, buffer in memories.items():
            tracker = getattr(buffer, "buffer_tracker", None)
            cache = getattr(tracker, "class_num_cache", None)
            if cache is None:
                continue
            cache_list = diagnostic_jsonable(cache)
            cache_values = np.array(cache_list).reshape(-1).tolist()
            if isinstance(cache_list, list):
                sizes[name] = {
                    "neg": int(cache_values[0]) if len(cache_values) > 0 else 0,
                    "pos": int(cache_values[1]) if len(cache_values) > 1 else 0,
                    "raw": cache_list,
                }
        return sizes

    def build_diagnostic_record(self, frame_index, result_dict, raw_dict):
        identifier = self.get_identifier()
        state = getattr(identifier, "state", None)
        state_name = state.state_name() if state is not None else None
        tracklets = getattr(identifier, "tracklets", {}) if identifier is not None else {}
        tracks_target_conf_bbox = raw_dict.get("tracks_target_conf_bbox") or {}
        bbox_score = raw_dict.get("bbox_score") or {}
        vis_indicator = raw_dict.get("vis_indicator") or {}
        vis_map = raw_dict.get("vis_map") or {}
        kpts = raw_dict.get("kpts") or {}
        ori = raw_dict.get("ori") or {}
        reliability_diagnostics = raw_dict.get("reliability_diagnostics") or {}
        association_decision = raw_dict.get("association_decision") or {}
        target_id = raw_dict.get("target_id", None)
        target_conf = raw_dict.get("target_conf", None)
        match_box = result_dict.get("match_box", None)
        candidate_ids = sorted(
            {
                int(key)
                for source in (tracks_target_conf_bbox, bbox_score, tracklets)
                for key in source.keys()
            }
        )

        candidates = {}
        selected_feature = None
        if target_id is not None and int(target_id) in tracklets:
            selected_feature = getattr(tracklets[int(target_id)], "deep_feature", None)

        for track_id in candidate_ids:
            tracklet = tracklets.get(track_id)
            track_conf_data = tracks_target_conf_bbox.get(track_id, tracks_target_conf_bbox.get(str(track_id)))
            part_scores = None
            agg_score = None
            bbox = None
            if track_conf_data is not None:
                part_scores, agg_score, bbox = track_conf_data
            if bbox is None:
                bbox_row = bbox_score.get(track_id, bbox_score.get(str(track_id)))
                if bbox_row is not None:
                    bbox = bbox_row[:4]

            kpt_values = kpts.get(track_id, kpts.get(str(track_id), []))
            pose_conf_values = []
            if isinstance(kpt_values, list):
                for item in kpt_values:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        pose_conf_values.append(diagnostic_float(item[2], 0.0))

            indicator = vis_indicator.get(track_id, vis_indicator.get(str(track_id), None))
            visible_count = None
            if isinstance(indicator, list):
                visible_count = int(sum(1 for value in indicator if bool(value)))
            reliability = reliability_diagnostics.get(track_id, reliability_diagnostics.get(str(track_id), {}))
            keypoint_conf = reliability.get("keypoint_confidence", {}) if isinstance(reliability, dict) else {}
            score_components = reliability.get("score_components", {}) if isinstance(reliability, dict) else {}

            visibility_map = vis_map.get(track_id, vis_map.get(str(track_id), None))
            visibility_map_summary = None
            if visibility_map is not None:
                arr = np.array(visibility_map)
                if arr.ndim == 3:
                    visibility_map_summary = {
                        "shape": list(arr.shape),
                        "part_pixel_sums": arr.reshape(arr.shape[0], -1).sum(axis=1).astype(int).tolist(),
                    }

            feature = getattr(tracklet, "deep_feature", None) if tracklet is not None else None
            feature_norms = None
            cosine_to_selected = None
            full_features = None
            if feature is not None:
                feature_cpu = feature.detach().cpu()
                feature_norms = {
                    "head": diagnostic_float(torch.norm(feature_cpu[0]).item()),
                    "torso": diagnostic_float(torch.norm(feature_cpu[1]).item()),
                    "legs": diagnostic_float(torch.norm(feature_cpu[2]).item()),
                    "feet": diagnostic_float(torch.norm(feature_cpu[3]).item()),
                    "global": diagnostic_float(torch.norm(feature_cpu[4]).item()),
                }
                if selected_feature is not None and int(track_id) != int(target_id):
                    cosine_to_selected = diagnostic_cosine(feature_cpu[4], selected_feature[4])
                if self.save_full_diagnostic_features:
                    full_features = feature_cpu.tolist()

            global_short_term_score = None
            if isinstance(part_scores, list):
                for global_index in (4, 9):
                    if global_index < len(part_scores) and part_scores[global_index] is not None:
                        global_short_term_score = diagnostic_float(part_scores[global_index])

            candidates[str(track_id)] = {
                "bbox": diagnostic_jsonable(bbox),
                "pose_confidence_mean": (
                    float(np.mean(pose_conf_values)) if pose_conf_values else None
                ),
                "pose_confidence_min": (
                    float(np.min(pose_conf_values)) if pose_conf_values else None
                ),
                "orientation": diagnostic_jsonable(ori.get(track_id, ori.get(str(track_id), None))),
                "visibility_indicator": diagnostic_jsonable(indicator),
                "visible_part_count": visible_count,
                "mean_keypoint_confidence": diagnostic_jsonable(keypoint_conf.get("mean")),
                "minimum_keypoint_confidence": diagnostic_jsonable(keypoint_conf.get("minimum")),
                "per_part_keypoint_confidence": diagnostic_jsonable(keypoint_conf.get("per_part")),
                "head_confidence": diagnostic_jsonable(keypoint_conf.get("head")),
                "torso_confidence": diagnostic_jsonable(keypoint_conf.get("torso")),
                "legs_confidence": diagnostic_jsonable(keypoint_conf.get("legs")),
                "feet_confidence": diagnostic_jsonable(keypoint_conf.get("feet")),
                "binary_orientation": diagnostic_jsonable(reliability.get("binary_orientation") if isinstance(reliability, dict) else None),
                "raw_orientation": diagnostic_jsonable(reliability.get("raw_orientation") if isinstance(reliability, dict) else None),
                "orientation_history": diagnostic_jsonable(reliability.get("orientation_history") if isinstance(reliability, dict) else None),
                "orientation_flip_count": diagnostic_jsonable(reliability.get("orientation_flip_count") if isinstance(reliability, dict) else None),
                "orientation_stability_score": diagnostic_jsonable(reliability.get("orientation_stability_score") if isinstance(reliability, dict) else None),
                "visibility_map_summary": visibility_map_summary,
                "feature_norms": feature_norms,
                "cosine_to_selected_global_feature": cosine_to_selected,
                "per_part_short_term_scores": diagnostic_jsonable(part_scores),
                "global_short_term_score": global_short_term_score,
                "active_fitted_classifier_slots": diagnostic_jsonable(score_components.get("active_fitted_classifier_slots") if isinstance(score_components, dict) else None),
                "active_front_bank_classifier_count": diagnostic_jsonable(score_components.get("active_front_bank_classifier_count") if isinstance(score_components, dict) else None),
                "active_back_bank_classifier_count": diagnostic_jsonable(score_components.get("active_back_bank_classifier_count") if isinstance(score_components, dict) else None),
                "global_classifier_available": diagnostic_jsonable(score_components.get("global_classifier_available") if isinstance(score_components, dict) else None),
                "current_part_average_score": diagnostic_jsonable(score_components.get("current_part_average_score") if isinstance(score_components, dict) else None),
                "current_global_score": diagnostic_jsonable(score_components.get("current_global_score") if isinstance(score_components, dict) else None),
                "current_final_score": diagnostic_jsonable(score_components.get("prototype_final_score") if isinstance(score_components, dict) else None),
                "front_bank_score": diagnostic_jsonable(score_components.get("front_bank_score") if isinstance(score_components, dict) else None),
                "back_bank_score": diagnostic_jsonable(score_components.get("back_bank_score") if isinstance(score_components, dict) else None),
                "orientation_reliability": diagnostic_jsonable(reliability.get("reliability_scores", {}).get("orientation_reliability") if isinstance(reliability, dict) else None),
                "pose_reliability": diagnostic_jsonable(reliability.get("reliability_scores", {}).get("pose_reliability") if isinstance(reliability, dict) else None),
                "selected_aggregation": diagnostic_jsonable(score_components.get("selected_aggregation") if isinstance(score_components, dict) else None),
                "long_term_model_scores": None,
                "long_term_model_scores_reason": "_reid_predict is implemented but not called by PartOCLWeightedClassifier.predict",
                "final_aggregated_appearance_score": diagnostic_float(agg_score),
                "full_features": full_features,
            }

        pairwise_global_cosine = {}
        for left_index, left_id in enumerate(candidate_ids):
            left_tracklet = tracklets.get(left_id)
            left_feature = getattr(left_tracklet, "deep_feature", None) if left_tracklet is not None else None
            for right_id in candidate_ids[left_index + 1:]:
                right_tracklet = tracklets.get(right_id)
                right_feature = getattr(right_tracklet, "deep_feature", None) if right_tracklet is not None else None
                if left_feature is not None and right_feature is not None:
                    pairwise_global_cosine[f"{left_id}:{right_id}"] = diagnostic_cosine(left_feature[4], right_feature[4])

        memory_sizes = self.get_diagnostic_memory_sizes(identifier) if identifier is not None else {}
        memory_changed = (
            self._last_diagnostic_memory_sizes is not None
            and memory_sizes != self._last_diagnostic_memory_sizes
        )
        self._last_diagnostic_memory_sizes = memory_sizes

        online_labels = {}
        if target_id is not None and int(target_id) >= 0:
            for track_id in candidate_ids:
                online_labels[str(track_id)] = 1 if int(track_id) == int(target_id) else 0

        classifier = getattr(identifier, "classifier", None) if identifier is not None else None
        params = getattr(classifier, "params", None)
        record = {
            "frame_index": int(frame_index),
            "candidate_track_ids": candidate_ids,
            "candidates": candidates,
            "pairwise_global_feature_cosine": pairwise_global_cosine,
            "target_lost_state": target_id is None or int(target_id) < 0,
            "state": state_name,
            "final_selected_track_id": diagnostic_jsonable(target_id),
            "final_selected_confidence": diagnostic_float(target_conf),
            "prediction_present": match_box is not None and target_id is not None and int(target_id) >= 0,
            "match_box": diagnostic_jsonable(match_box),
            "online_memory_insertion_event_inferred": bool(memory_changed),
            "online_memory_sizes": memory_sizes,
            "assigned_online_labels_if_memory_updated": online_labels,
            "online_model_update_event_inferred": {
                "newest_st_loss": diagnostic_float(getattr(identifier, "newest_st_loss", None)) if identifier is not None else None,
                "newest_lt_loss": diagnostic_float(getattr(identifier, "newest_lt_loss", None)) if identifier is not None else None,
                "incremental_st_loss": diagnostic_jsonable(getattr(identifier, "incremental_st_loss", None)) if identifier is not None else None,
            },
            "thresholds": {
                "select_target_iou": diagnostic_float(getattr(self.hyper_config, "select_target_threshold", None)),
                "id_switch_detection": diagnostic_float(getattr(params, "id_switch_detection_thresh", None)) if params is not None else None,
                "reid_pos_confidence": diagnostic_float(getattr(params, "reid_pos_confidence_thresh", None)) if params is not None else None,
                "min_target_confidence": diagnostic_float(getattr(params, "min_target_confidence", None)) if params is not None else None,
                "reliability_mode": getattr(params, "reliability_mode", "baseline") if params is not None else "baseline",
                "association_mode": getattr(params, "association_mode", "baseline") if params is not None else "baseline",
                "association_reid_threshold": diagnostic_float(getattr(params, "association_reid_threshold", None)) if params is not None else None,
                "association_reid_margin": diagnostic_float(getattr(params, "association_reid_margin", None)) if params is not None else None,
            },
            "association_decision": diagnostic_jsonable(association_decision),
            "selection_rejection_reason": self.diagnostic_selection_reason(target_id, target_conf, candidate_ids, state_name),
        }
        return diagnostic_jsonable(record)

    def diagnostic_selection_reason(self, target_id, target_conf, candidate_ids, state_name):
        if not candidate_ids:
            return "no_candidate_track"
        if target_id is None:
            return "target_not_initialized"
        if int(target_id) < 0:
            return "reidentification_state_no_track_confirmed"
        conf = diagnostic_float(target_conf)
        if state_name == "re-identification":
            return "selected_track_not_yet_reconfirmed_by_consecutive_reid_threshold"
        if conf is None:
            return "selected_track_has_no_appearance_score"
        return "selected_track_reported_by_identifier_state"

    def write_diagnostic_record(self, frame_index, result_dict, raw_dict):
        if self.diagnostic_log is None:
            return
        self.diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
        record = self.build_diagnostic_record(frame_index, result_dict, raw_dict)
        with self.diagnostic_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def get_runtime_reid_model(self):
        if self.tracker is None:
            return None
        obj_tracker = getattr(self.tracker, "obj_tracker", None)
        return getattr(obj_tracker, "reid", None)

    def hash_runtime_reid(self, keys=None):
        reid_model = self.get_runtime_reid_model()
        if reid_model is None:
            return None, {}

        state = reid_model.state_dict()
        if keys is not None:
            state = {
                key: state[key]
                for key in keys
                if key in state
            }
        return hash_tensors(state), state

    def log_reid_initial_hashes(self):
        reid_model = self.get_runtime_reid_model()
        if reid_model is None:
            print("[REID RESET] No runtime ReID model found for hashing.")
            return

        checkpoint_hash, checkpoint_tensors = checkpoint_tensor_hash(
            self.reid_checkpoint
        )
        runtime_state = reid_model.state_dict()
        self.reid_loaded_keys = [
            key
            for key, value in checkpoint_tensors.items()
            if key in runtime_state
            and tuple(value.shape) == tuple(runtime_state[key].shape)
        ]
        loaded_subset = {
            key: runtime_state[key]
            for key in self.reid_loaded_keys
        }

        self.reid_checkpoint_hash = checkpoint_hash
        self.reid_initial_loaded_hash = hash_tensors(loaded_subset)
        self.reid_initial_full_hash, full_state = self.hash_runtime_reid()

        print(f"[REID RESET] checkpoint_path={self.reid_checkpoint.resolve()}")
        print(f"[REID RESET] checkpoint_hash={self.reid_checkpoint_hash}")
        print(f"[REID RESET] checkpoint_key_count={len(checkpoint_tensors)}")
        print(f"[REID RESET] matched_loaded_key_count={len(self.reid_loaded_keys)}")
        print(f"[REID RESET] initial_loaded_hash={self.reid_initial_loaded_hash}")
        print(f"[REID RESET] initial_full_reid_hash={self.reid_initial_full_hash}")
        print(
            "[REID RESET] initial_loaded_hash_matches_checkpoint="
            f"{self.reid_initial_loaded_hash == self.reid_checkpoint_hash}"
        )
        print(f"[REID RESET] full_reid_key_count={len(full_state)}")

    def log_reid_final_hashes(self):
        if not self.reid_loaded_keys:
            return

        final_loaded_hash, _ = self.hash_runtime_reid(self.reid_loaded_keys)
        final_full_hash, _ = self.hash_runtime_reid()
        print(f"[REID RESET] final_loaded_hash={final_loaded_hash}")
        print(f"[REID RESET] final_full_reid_hash={final_full_hash}")
        print(
            "[REID RESET] loaded_params_changed_during_sequence="
            f"{final_loaded_hash != self.reid_initial_loaded_hash}"
        )
        print(
            "[REID RESET] full_reid_changed_during_sequence="
            f"{final_full_hash != self.reid_initial_full_hash}"
        )

    def disable_live_display(self, reason):
        if self.live_display_enabled:
            print(f"\n[WARNING] Live display disabled: {reason}")
        self.live_display_enabled = False
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def handle_live_keyboard(self):
        while True:
            key = cv2.waitKey(0 if self.live_paused else 1) & 0xFF
            if key in (255,):
                return
            if key == ord(' '):
                self.live_paused = not self.live_paused
                if not self.live_paused:
                    return
                continue
            if key in (ord('q'), ord('Q'), 27):
                self.disable_live_display("window closed by user; continuing inference headlessly")
                return
            if not self.live_paused:
                return

    def draw_live_overlay(self, img_disp, frame_index, target_id, target_conf):
        target_lost = target_id is None or int(target_id) < 0
        lines = [
            f"Method: {self.method}",
            f"Video frame: {frame_index}",
            f"Selected tracker ID: {target_id if not target_lost else 'None'}",
            f"Selected target confidence: {float(target_conf):.3f}" if target_conf is not None else "Selected target confidence: -1.000",
        ]
        if target_lost:
            lines.append("TARGET LOST")

        x = 12
        y = 28
        for line in lines:
            color = (0, 0, 255) if line == "TARGET LOST" else (255, 255, 255)
            cv2.putText(
                img_disp,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_disp,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            y += 26

    def get_input_fps(self):
        if self.input is None or osp.isdir(self.input):
            return 30.0

        capture = cv2.VideoCapture(self.input)
        if not capture.isOpened():
            return 30.0

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()

        if fps <= 0 or math.isnan(fps):
            return 30.0

        return fps

    def init_visualization_writer(self, frame):
        if not self.visualization_writer_enabled or self.visualization_writer is not None:
            return

        try:
            parent = osp.dirname(osp.abspath(self.visualization_video))
            os.makedirs(parent, exist_ok=True)

            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                self.visualization_video,
                fourcc,
                self.visualization_fps,
                (width, height),
            )

            if not writer.isOpened():
                writer.release()
                print(
                    f"\n[WARNING] Could not open visualization video writer: "
                    f"{self.visualization_video}. Continuing without MP4 saving."
                )
                self.visualization_writer_enabled = False
                return

            self.visualization_writer = writer
            print(
                f"\n[OK] Saving inference visualization MP4: "
                f"{self.visualization_video}"
            )

        except Exception as error:
            print(
                f"\n[WARNING] Could not initialise visualization video writer: "
                f"{error}. Continuing without MP4 saving."
            )
            self.visualization_writer_enabled = False

    def release_visualization_writer(self):
        if self.visualization_writer is not None:
            self.visualization_writer.release()
            self.visualization_writer = None
    
    def init_work_seed(self, seed=123):
        """Sets random seeds for reproducibility across all libraries."""
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        sklearn.utils.check_random_state(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_tracker(self, seed):
        """Initializes the tracker and creates requested output directories."""
        if self.output is not None:
            os.makedirs(self.output, exist_ok=True)

        if self.output_json is not None:
            json_parent = osp.dirname(osp.abspath(self.output_json))
            os.makedirs(json_parent, exist_ok=True)

        return Tracker(
            self.type,
            self.config,
            self.ckpt,
            hyper_config=self.hyper_config,
            seed=seed,
            identifier_config=self.identifier_params,
        )

    def run_video(self):
        """Main loop for video/image sequence processing."""
        if self.show_result and not graphical_display_available():
            self.disable_live_display("no graphical display detected; continuing headlessly")

        # Load images from directory or video file
        if osp.isdir(self.input):
            imgs = sorted(
                filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')), os.listdir(self.input)),
                key=lambda x: int(x.split('.')[0]))
        else:
            imgs = mmcv.VideoReader(self.input)

        self.init_work_seed(self.seed)
        self.tracker = self.init_tracker(self.seed)
        self.log_reid_initial_hashes()
        self.visualization_fps = self.get_input_fps()

        prog_bar = mmcv.ProgressBar(len(imgs))

        try:
            for i, img in enumerate(imgs):
                # Skip frames until reaching the start_frame
                if i < self.start_frame:
                    prog_bar.update()
                    continue

                # Handle image loading based on input type
                if isinstance(img, str):
                    img_name = os.path.splitext(img)[0]
                    img_path = osp.join(self.input, img)
                    img = mmcv.imread(img_path)
                else:
                    img_name = f'{i:06d}.jpg'

                # Target selection (ROI) on the starting frame
                if i == self.start_frame:
                    if self.gt_bbox_file is not None:
                        if not osp.isfile(self.gt_bbox_file):
                            raise FileNotFoundError(
                                f"Initial bounding-box file not found: {self.gt_bbox_file}"
                            )

                        bboxes = [
                            line.strip()
                            for line in mmcv.list_from_file(self.gt_bbox_file)
                            if line.strip()
                        ]

                        if not bboxes:
                            raise ValueError(
                                f"Initial bounding-box file is empty: {self.gt_bbox_file}"
                            )

                        init_bbox = list(map(float, bboxes[0].split(',')))
                    else:
                        # Dynamic window title to identify the current frame
                        window_name = f"Select Target - Frame {i}"
                        init_bbox = list(cv2.selectROI(window_name, img, False, False))
                        cv2.destroyWindow(window_name)

                    if len(init_bbox) != 4:
                        raise ValueError(
                            "The initial bounding box must contain exactly four values: "
                            "x,y,width,height"
                        )

                    if init_bbox[2] <= 0 or init_bbox[3] <= 0:
                        raise ValueError(
                            "The initial bounding box must have positive width and height."
                        )

                    # Convert (x, y, w, h) to (x1, y1, x2, y2)
                    init_bbox[2] += init_bbox[0]
                    init_bbox[3] += init_bbox[1]
                else:
                    init_bbox = None

                # AI Inference call
                result_dict, raw_dict = self.tracker.infer(img, init_bbox, i)
                self.write_diagnostic_record(i, result_dict, raw_dict)

                # Extract data from inference results
                match_box = result_dict.get("match_box", None)
                target_id = raw_dict.get("target_id", -1)
                target_conf = raw_dict.get("target_conf", -1)
                det_bboxes = raw_dict.get("det_bboxes", None)
                tracks_target_conf_bbox = raw_dict.get("tracks_target_conf_bbox", None)
                threshold = raw_dict.get("threshold", None)

                if target_conf is None:
                    target_conf = -1

                # --- OPTIMIZATION: RECORD DATA TO MEMORY (DICTIONARY) ---
                if self.output_json is not None:
                    self.result[f'{img_name}'] = {}
                    self.result[f'{img_name}']['target_info'] = [target_id] + match_box + [target_conf] if match_box is not None else [target_id, 0, 0, 0, 0, target_conf]
                    
                    if det_bboxes is not None:
                        self.result[f'{img_name}']['det_bboxes'] = det_bboxes[0].tolist()
                    
                    self.result[f'{img_name}']['tracks_target_conf_bbox'] = tracks_target_conf_bbox
                    
                    if threshold is not None:
                        self.result[f'{img_name}']['threshold'] = threshold

                    # --- OPTIMIZATION: BATCH SAVE TO DISK EVERY 200 FRAMES ---
                    if i % 200 == 0:
                        write_to_json(self.output_json, self.result)

                # Drawing and Visualization
                img_disp = img
                if (
                    self.show_result
                    or self.output is not None
                    or self.visualization_writer_enabled
                ):
                    img_disp = img.copy()
                    if tracks_target_conf_bbox is not None:
                        for track_id in tracks_target_conf_bbox.keys():
                            _, t_conf, track_bbox = tracks_target_conf_bbox[track_id]
                            if t_conf is None:
                                t_conf = -1
                            # Green for target, Red for others
                            color = (0, 255, 0) if track_id == target_id else (0, 0, 255)
                            cv2.rectangle(img_disp, (int(track_bbox[0]), int(track_bbox[1])), 
                                          (int(track_bbox[2]), int(track_bbox[3])), color, 3)
                            cv2.putText(img_disp, f'ID:{track_id}, {t_conf:.2f}', (int(track_bbox[0])+10, int(track_bbox[1])+30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    self.draw_live_overlay(
                        img_disp,
                        i,
                        target_id,
                        target_conf,
                    )

                if self.visualization_writer_enabled:
                    self.init_visualization_writer(img_disp)
                    if self.visualization_writer is not None:
                        try:
                            self.visualization_writer.write(img_disp)
                        except cv2.error as error:
                            print(
                                f"\n[WARNING] Could not write visualization "
                                f"frame {i}: {error}. Continuing without "
                                "MP4 saving."
                            )
                            self.release_visualization_writer()
                            self.visualization_writer_enabled = False

                # Display frame window
                if self.live_display_enabled:
                    try:
                        window_title = (
                            f"OCLReID | {self.method} | "
                            f"Frame {i:06d} | Target "
                            f"{target_id if target_id is not None and int(target_id) >= 0 else 'LOST'}"
                        )
                        cv2.namedWindow(self.live_window_name, cv2.WINDOW_NORMAL)
                        if hasattr(cv2, "setWindowTitle"):
                            cv2.setWindowTitle(self.live_window_name, window_title)
                        cv2.imshow(self.live_window_name, img_disp)
                        self.handle_live_keyboard()
                    except cv2.error as error:
                        self.disable_live_display(
                            f"OpenCV window error ({error}); continuing headlessly"
                        )

                # Save processed frame to disk if output path is provided
                if self.output is not None:
                    cv2.imwrite(os.path.join(self.output, f'{i:06d}.jpg'), img_disp)

                prog_bar.update()

            # --- FINAL SAVE: Ensure the last processed frames are saved to disk ---
            if self.output_json is not None:
                write_to_json(self.output_json, self.result)
                print(f"\n[OK] Final results saved to: {self.output_json}")

        finally:
            self.log_reid_final_hashes()
            self.release_visualization_writer()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

if __name__ == '__main__':
    parser = ArgumentParser(description='OCL-ReID Evaluator Script')
    parser.add_argument('--input', type=str, help='Path to input video or image directory', default=None)
    parser.add_argument('--output', type=str, default=None, help='Path to save processed images')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save results in JSON format')
    parser.add_argument('--visualization-video', type=str, default=None, help='Path to save annotated inference as an MP4')
    parser.add_argument('--show_result', action='store_true', help='Whether to display tracking results in a window')
    parser.add_argument('--reid-checkpoint', type=str, default=str(DEFAULT_REID_CHECKPOINT), help='Initial ReID checkpoint to load')
    parser.add_argument('--diagnostic-log', type=str, default=None, help='Optional JSONL path for per-frame diagnostic logging')
    parser.add_argument('--save-full-diagnostic-features', action='store_true', help='Store full 512-D part/global embeddings in diagnostic JSONL')
    parser.add_argument('--reliability-mode', choices=['baseline', 'global_fallback', 'dual_orientation', 'combined'], default='baseline')
    parser.add_argument('--pose-confidence-threshold', type=float, default=0.0)
    parser.add_argument('--minimum-visible-parts', type=int, default=0)
    parser.add_argument('--orientation-history-length', type=int, default=5)
    parser.add_argument('--orientation-flip-threshold', type=int, default=2)
    parser.add_argument('--minimum-reliability', type=float, default=0.0)
    parser.add_argument('--global-fallback-weight', type=float, default=0.0)
    parser.add_argument('--dual-orientation-aggregation', choices=['min', 'mean', 'max'], default='min')
    parser.add_argument('--association-mode', choices=['baseline', 'reid_gate'], default='baseline')
    parser.add_argument('--association-reid-threshold', type=float, default=0.6)
    parser.add_argument('--association-reid-margin', type=float, default=0.02)
    parser.add_argument('--association-min-bbox-score', type=float, default=0.0)
    parser.add_argument('--association-min-visible-parts', type=int, default=1)
    parser.add_argument('--start_frame', type=int, default=0, help='Frame number to start tracking from')
    parser.add_argument(
        '--gt_bbox_file',
        type=str,
        default=None,
        help='Optional initial target bbox file containing x,y,width,height',
    )
    parser.add_argument('--method', type=str, choices=['part-OCLReID', 'global-OCLReID', 'rpf-ReID'], default='part-OCLReID')
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=384)
    args = parser.parse_args()

    method = args.method
    # TPT configurations directory mapping
    base_dir = osp.join(PROJECT_ROOT, "tpt_configs")
    to_be_runned = {
        "rpf-ReID": "baseline_oclreid_resnet18.py",
        "global-OCLReID": "baseline_oclreid_finenued_global_resnet18.py",
        "part-OCLReID": "baseline_oclreid_finenued_resnet18.py",
    }

    print(f"\nRunning: {method}\n")
    hyper_params = Config.fromfile(osp.join(base_dir, to_be_runned[method]))
    hyper_params.mmtracking_dir = PROJECT_ROOT
    
    # Use demo video if no input is provided
    if args.input is None:
        #args.input = osp.join(PROJECT_ROOT, "demo_video.mp4")
        args.input = osp.join(PROJECT_ROOT, "cam_zed_rgb.mp4")
    
    # Map command line arguments to hyper_params config
    hyper_params.input = args.input
    hyper_params.output = args.output
    hyper_params.output_json = args.output_json
    hyper_params.visualization_video = args.visualization_video
    hyper_params.show_result = args.show_result
    hyper_params.diagnostic_log = args.diagnostic_log
    hyper_params.save_full_diagnostic_features = args.save_full_diagnostic_features
    hyper_params.method = args.method
    hyper_params.start_frame = args.start_frame
    hyper_params.gt_bbox_file = args.gt_bbox_file
    hyper_params.image_shape = (args.img_width, args.img_height, 3)

    # Setup RPF config and ReID checkpoint path
    rpf_cfg_path = osp.join(hyper_params.mmtracking_dir, hyper_params.rpf_config)
    rpf_config = mmcv.Config.fromfile(rpf_cfg_path)
    reid_checkpoint = Path(args.reid_checkpoint).expanduser()
    if not reid_checkpoint.is_absolute():
        reid_checkpoint = PROJECT_ROOT / reid_checkpoint
    reid_checkpoint = reid_checkpoint.resolve()
    if not reid_checkpoint.exists():
        raise FileNotFoundError(f"ReID checkpoint not found: {reid_checkpoint}")
    print(f"ReID checkpoint: {reid_checkpoint}")
    hyper_params.reid_checkpoint = str(reid_checkpoint)
    rpf_config.model.reid.init_cfg.checkpoint = str(reid_checkpoint)
    
    # Load identifier configuration
    identifier_config = osp.join(hyper_params.mmtracking_dir, hyper_params.identifier_config)
    identifier_preview = Config.fromfile(identifier_config)
    identifier_preview.identifier_params.params.reliability_mode = args.reliability_mode
    identifier_preview.identifier_params.params.pose_confidence_threshold = args.pose_confidence_threshold
    identifier_preview.identifier_params.params.minimum_visible_parts = args.minimum_visible_parts
    identifier_preview.identifier_params.params.orientation_history_length = args.orientation_history_length
    identifier_preview.identifier_params.params.orientation_flip_threshold = args.orientation_flip_threshold
    identifier_preview.identifier_params.params.minimum_reliability = args.minimum_reliability
    identifier_preview.identifier_params.params.global_fallback_weight = args.global_fallback_weight
    identifier_preview.identifier_params.params.dual_orientation_aggregation = args.dual_orientation_aggregation
    identifier_preview.identifier_params.params.association_mode = args.association_mode
    identifier_preview.identifier_params.params.association_reid_threshold = args.association_reid_threshold
    identifier_preview.identifier_params.params.association_reid_margin = args.association_reid_margin
    identifier_preview.identifier_params.params.association_min_bbox_score = args.association_min_bbox_score
    identifier_preview.identifier_params.params.association_min_visible_parts = args.association_min_visible_parts
    tmp_identifier_config = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    )
    try:
        tmp_identifier_config.write(identifier_preview.pretty_text)
        tmp_identifier_config.close()

    # Initialize and run the evaluator
        evaluator = TargetIdentificationEvaluator(hyper_params, rpf_config, tmp_identifier_config.name)
        evaluator.run_video()
    finally:
        try:
            os.unlink(tmp_identifier_config.name)
        except OSError:
            pass
