#!/usr/bin/env python3
"""
Duplicate Photo & Video Finder - Flask Web App
Scans for duplicate media files and provides a web UI to review and delete them.
"""

import os
import sys
import json
import hashlib
import subprocess
import threading
import webbrowser
import time
from collections import defaultdict
from datetime import datetime
from io import BytesIO

from flask import Flask, jsonify, request, send_from_directory, Response
from PIL import Image, ImageFile

# Allow loading of truncated/corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm", ".3gp"}
ALL_MEDIA_EXTS = PHOTO_EXTS | VIDEO_EXTS

# Data directory - configurable for Docker deployments
DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

# File paths - use DATA_DIR for all persistent data
THUMB_DIR = os.path.join(DATA_DIR, "thumbnails")
THUMB_SIZE = (150, 150)
THUMB_QUALITY = 50
HISTORY_FILE = os.path.join(DATA_DIR, "deletion_history.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
PARTIAL_HASH_SIZE = 8192  # 8 KB for partial hashing

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Failed to save settings: {e}")

# Priority order for ROOT_DIR: ENV > settings.json > default
_settings = load_settings()
ROOT_DIR = os.getenv("ROOT_DIR")  # ENV variable takes precedence
if not ROOT_DIR:
    ROOT_DIR = _settings.get("root_dir", "~/Pictures")  # Fallback to settings.json
ROOT_DIR = os.path.expanduser(ROOT_DIR)

# Ensure ROOT_DIR exists
if not os.path.isdir(ROOT_DIR):
    fallback = os.path.expanduser("~/Pictures")
    print(f"Warning: ROOT_DIR '{ROOT_DIR}' does not exist. Falling back to {fallback}")
    ROOT_DIR = fallback
ROOT_DIR = os.path.abspath(ROOT_DIR)

# Priority order for PORT: ENV > default
PORT = int(os.getenv("PORT", "8080"))

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
# duplicates: {hash: [{"path": ..., "size": ..., "type": ...}, ...]}
duplicates = {}
all_media_files = []
scan_complete = False
scan_stats = {
    "total_files_scanned": 0,
    "phase": "idle",
    "phase_detail": "",
}

# ---------------------------------------------------------------------------
# Deletion history persistence
# ---------------------------------------------------------------------------

def load_history():
    history = {
        "total_space_saved": 0,
        "total_images_deleted": 0,
        "total_videos_deleted": 0,
        "space_saved_images": 0,
        "space_saved_videos": 0,
        "deletions": []
    }
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                history.update(data)
                
                # Backfill if necessary
                if history["total_space_saved"] > 0 and history["total_images_deleted"] == 0 and history["total_videos_deleted"] == 0:
                    print("Migrating history for granular stats...")
                    for d in history.get("deletions", []):
                        path_str = d.get("path", "")
                        ext = os.path.splitext(path_str)[1].lower()
                        if ext in VIDEO_EXTS:
                            history["total_videos_deleted"] += 1
                            history["space_saved_videos"] += d.get("size", 0)
                        else:
                            history["total_images_deleted"] += 1
                            history["space_saved_images"] += d.get("size", 0)
        except (json.JSONDecodeError, IOError):
            pass
    return history


def enrich_history_from_index():
    """Use all_files_index.json to find 'kept' versions for old history entries."""
    global deletion_history
    index_path = os.path.join(DATA_DIR, "all_files_index.json")
    if not os.path.exists(index_path):
        return 0
        
    try:
        with open(index_path, 'r') as f:
            all_files = json.load(f)
    except Exception:
        return 0
        
    # size -> [paths]
    size_map = defaultdict(list)
    for f in all_files:
        size_map[f['size']].append(f['path'])
        
    updated_count = 0
    print(f"Enriching history using {len(all_files)} indexed files...")
    for entry in deletion_history.get("deletions", []):
        size = entry.get("size")
        path = entry.get("path")
        
        # If path is missing, try to guess from index
        if not path and size and size in size_map:
            entry["path"] = size_map[size][0]
            path = entry["path"]
            updated_count += 1

        if "kept_versions" not in entry or not entry["kept_versions"]:
            if size and size in size_map:
                others = [p for p in size_map[size] if p != path]
                if others:
                    entry["kept_versions"] = others
                    updated_count += 1
                    
    if updated_count > 0:
        save_history(deletion_history)
        print(f"  Successfully enriched {updated_count} history entries.")
    else:
        print("  No history entries needed enrichment.")
    return updated_count


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


deletion_history = load_history()

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def is_within_root(path):
    """Security check: ensure the path is under ROOT_DIR."""
    try:
        real = os.path.realpath(path)
        root = os.path.realpath(ROOT_DIR)
        return real.startswith(root + os.sep) or real == root
    except Exception:
        return False


def get_thumb_filename(path):
    path_hash = hashlib.md5(path.encode()).hexdigest()[:12]
    return f"{path_hash}.jpg"

# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

def make_thumbnail(path):
    """Create a JPEG thumbnail. Returns the filename on success, None otherwise."""
    try:
        thumb_filename = get_thumb_filename(path)
        thumb_path = os.path.join(THUMB_DIR, thumb_filename)

        if os.path.exists(thumb_path):
            return thumb_filename

        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTS:
            return _make_video_thumbnail(path, thumb_filename)

        with Image.open(path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.thumbnail(THUMB_SIZE)
            img.save(thumb_path, format="JPEG", quality=THUMB_QUALITY, optimize=True)
            return thumb_filename
    except Exception:
        return None


def _make_video_thumbnail(path, thumb_filename):
    try:
        thumb_path = os.path.join(THUMB_DIR, thumb_filename)
        ffmpeg_quality = max(2, min(15, int((100 - THUMB_QUALITY) / 7) + 2))
        
        # Try at 1 second first
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-ss", "00:00:01",
            "-vframes", "1",
            "-vf", f"scale={THUMB_SIZE[0]}:{THUMB_SIZE[1]}:force_original_aspect_ratio=decrease",
            "-q:v", str(ffmpeg_quality),
            thumb_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=10)
        
        # If it failed (maybe video is < 1s), try at 0s
        if not os.path.exists(thumb_path) or os.path.getsize(thumb_path) == 0:
            cmd[cmd.index("-ss") + 1] = "00:00:00"
            subprocess.run(cmd, capture_output=True, timeout=10)

        if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
            return thumb_filename
        if os.path.exists(thumb_path):
            os.unlink(thumb_path)
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 3-phase duplicate scan
# ---------------------------------------------------------------------------

def _partial_hash(path):
    """Hash the first and last PARTIAL_HASH_SIZE bytes of a file."""
    h = hashlib.sha256()
    try:
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(PARTIAL_HASH_SIZE)
            h.update(head)
            if file_size > PARTIAL_HASH_SIZE * 2:
                f.seek(-PARTIAL_HASH_SIZE, 2)
                tail = f.read(PARTIAL_HASH_SIZE)
                h.update(tail)
            elif file_size > PARTIAL_HASH_SIZE:
                f.seek(-PARTIAL_HASH_SIZE, 2)
                tail = f.read()
                h.update(tail)
    except Exception:
        return None
    return h.hexdigest()


def _full_hash(path):
    """Full SHA-256 of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except Exception:
        return None
    return h.hexdigest()


def run_scan():
    """3-phase scan: size grouping -> partial hash -> full hash."""
    global duplicates, all_media_files, scan_complete, scan_stats

    scan_stats["phase"] = "phase1"
    scan_stats["phase_detail"] = "Grouping files by size..."
    print("Phase 1: Grouping files by size...")

    size_groups = defaultdict(list)  # size -> [{"path", "size", "type"}]
    total_scanned = 0
    all_files = [] # Track all media files found

    for dirpath, _, filenames in os.walk(ROOT_DIR):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in ALL_MEDIA_EXTS:
                continue
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
                file_size = st.st_size
                if file_size == 0:
                    continue
                file_type = "video" if ext in VIDEO_EXTS else "photo"
                size_groups[file_size].append({"path": path, "size": file_size, "type": file_type})
                total_scanned += 1
                if total_scanned % 1000 == 0:
                    scan_stats["phase_detail"] = f"Indexed {total_scanned} files..."
                
                # Add to full index
                all_files.append({
                    "path": path,
                    "size": file_size,
                    "type": file_type,
                    "modified": st.st_mtime
                })
            except OSError:
                continue

    scan_stats["total_files_scanned"] = total_scanned
    # Keep only sizes with 2+ files
    size_groups = {s: files for s, files in size_groups.items() if len(files) > 1}
    candidates_after_p1 = sum(len(v) for v in size_groups.values())
    print(f"  {total_scanned} files scanned, {candidates_after_p1} candidates in {len(size_groups)} size groups")

    # Save full file index
    all_files_path = os.path.join(DATA_DIR, "all_files_index.json")
    try:
        with open(all_files_path, 'w') as f:
            json.dump(all_files, f, indent=2)
        print(f"  Saved full file index to {all_files_path}")
        all_media_files = all_files
    except Exception as e:
        print(f"  Failed to save file index: {e}")

    # Phase 2 – partial hash
    scan_stats["phase"] = "phase2"
    scan_stats["phase_detail"] = "Partial hashing candidates..."
    print("Phase 2: Partial hashing candidates...")

    partial_groups = defaultdict(list)  # partial_hash -> [file_info]
    hashed = 0
    for file_list in size_groups.values():
        for fi in file_list:
            ph = _partial_hash(fi["path"])
            if ph:
                partial_groups[ph].append(fi)
            hashed += 1
            if hashed % 500 == 0:
                scan_stats["phase_detail"] = f"Partial-hashed {hashed}/{candidates_after_p1}..."

    partial_groups = {h: files for h, files in partial_groups.items() if len(files) > 1}
    candidates_after_p2 = sum(len(v) for v in partial_groups.values())
    print(f"  {candidates_after_p2} candidates remain after partial hash")

    # Phase 3 – full hash
    scan_stats["phase"] = "phase3"
    scan_stats["phase_detail"] = "Full hashing remaining candidates..."
    print("Phase 3: Full hashing remaining candidates...")

    full_groups = defaultdict(list)
    hashed = 0
    for file_list in partial_groups.values():
        for fi in file_list:
            fh = _full_hash(fi["path"])
            if fh:
                full_groups[fh].append(fi)
            hashed += 1
            if hashed % 200 == 0:
                scan_stats["phase_detail"] = f"Full-hashed {hashed}/{candidates_after_p2}..."

    duplicates = {h: files for h, files in full_groups.items() if len(files) > 1}
    print(f"  Found {len(duplicates)} duplicate groups")

    # Generate thumbnails
    scan_stats["phase"] = "thumbnails"
    total_thumb_files = sum(len(files) for files in duplicates.values())
    scan_stats["phase_detail"] = f"Generating thumbnails for {total_thumb_files} files..."
    print(f"Generating thumbnails for {total_thumb_files} files...")

    done = 0
    for files in duplicates.values():
        for fi in files:
            make_thumbnail(fi["path"])
            done += 1
            if done % 50 == 0:
                scan_stats["phase_detail"] = f"Thumbnails: {done}/{total_thumb_files}"
                print(f"  Thumbnails: {done}/{total_thumb_files}")

    scan_stats["phase"] = "done"
    scan_stats["phase_detail"] = "Scan complete."
    scan_complete = True
    print("Scan complete. Server is ready.")

# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats():
    total_groups = len(duplicates)
    total_files = sum(len(files) for files in duplicates.values())
    total_photos = 0
    total_videos = 0
    wasted = 0
    folder_set = set()
    for files in duplicates.values():
        wasted += files[0]["size"] * (len(files) - 1)
        for f in files:
            folder_set.add(os.path.dirname(f["path"]))
            if f["type"] == "video":
                total_videos += 1
            else:
                total_photos += 1
    return {
        "total_groups": total_groups,
        "total_files": total_files,
        "total_photos": total_photos,
        "total_videos": total_videos,
        "wasted_space": wasted,
        "wasted_space_human": format_size(wasted),
        "folders_with_duplicates": len(folder_set),
        "total_files_scanned": scan_stats["total_files_scanned"],
        "space_saved": deletion_history["total_space_saved"],
        "space_saved_human": format_size(deletion_history["total_space_saved"]),
        "images_deleted": deletion_history.get("total_images_deleted", 0),
        "videos_deleted": deletion_history.get("total_videos_deleted", 0),
        "space_saved_images_human": format_size(deletion_history.get("space_saved_images", 0)),
        "space_saved_videos_human": format_size(deletion_history.get("space_saved_videos", 0)),
        "scan_complete": scan_complete,
        "scan_phase": scan_stats["phase"],
        "scan_detail": scan_stats["phase_detail"],
    }


def compute_folder_stats():
    # Only count files that have at least one duplicate in a DIFFERENT folder
    folder_cross_duplicates = defaultdict(lambda: {"count": 0, "size": 0})
    
    for h, files in duplicates.items():
        folders_in_group = set(os.path.dirname(f["path"]) for f in files)
        if len(folders_in_group) > 1:
            # This group spans multiple folders, so it's relevant for folder-level cleaning
            for f in files:
                folder = os.path.dirname(f["path"])
                folder_cross_duplicates[folder]["count"] += 1
                folder_cross_duplicates[folder]["size"] += f["size"]
                
    result = []
    for folder, info in sorted(folder_cross_duplicates.items(), key=lambda x: -x[1]["count"]):
        result.append({
            "folder": folder,
            "count": info["count"],
            "size": info["size"],
            "size_human": format_size(info["size"]),
        })
    return result


def move_to_trash(path):
    """Move a file to Trash. Returns (success: bool, error: str|None)."""
    if not os.path.exists(path):
        return False, "File not found"
    if not is_within_root(path):
        return False, "Path is outside the allowed root directory"
    try:
        import send2trash
        send2trash.send2trash(path)
        return True, None
    except Exception as e1:
        # Fallback: macOS osascript
        try:
            escaped = path.replace("\\", "\\\\").replace('"', '\\"')
            script = f'tell application "Finder" to delete POSIX file "{escaped}"'
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=10)
            if not os.path.exists(path):
                return True, None
            return False, "osascript ran but file still exists"
        except Exception as e2:
            return False, f"send2trash: {e1}; osascript: {e2}"


def remove_file_from_duplicates(path):
    """Remove a file entry from the in-memory duplicates dict.
    If a group drops below 2, remove the group entirely.
    Returns True if found and removed."""
    for h, files in list(duplicates.items()):
        for i, fi in enumerate(files):
            if fi["path"] == path:
                files.pop(i)
                if len(files) < 2:
                    del duplicates[h]
                return True
    return False


def remove_file_from_index(path):
    """Remove a file from the global media index."""
    global all_media_files
    for i, f in enumerate(all_media_files):
        if f["path"] == path:
            all_media_files.pop(i)
            return True
    return False

# ---------------------------------------------------------------------------
# Empty folders
# ---------------------------------------------------------------------------

def is_effectively_empty(path):
    """Checks if a directory is empty or only contains .DS_Store."""
    try:
        items = os.listdir(path)
        for item in items:
            if item != ".DS_Store":
                return False
        return True
    except OSError:
        return False


def get_empty_folders():
    """Finds all folders that contain no files and no non-empty subfolders."""
    empty_folders = []
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR, topdown=False):
        # Skip ROOT_DIR itself
        if os.path.realpath(dirpath) == os.path.realpath(ROOT_DIR):
            continue
            
        if is_effectively_empty(dirpath):
            empty_folders.append(dirpath)
    return sorted(empty_folders)


# ---------------------------------------------------------------------------
# Semantic Search (CLIP)
# ---------------------------------------------------------------------------

class SemanticSearch:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.embeddings = {}  # path -> embedding (list)
        self.embeddings_file = os.path.join(DATA_DIR, "image_embeddings.json")
        self.is_loading = False
        self.is_indexing = False
        self.index_progress = {"current": 0, "total": 0, "status": "idle"}
        self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, "r") as f:
                    self.embeddings = json.load(f)
            except Exception:
                self.embeddings = {}

    def save_embeddings(self):
        try:
            with open(self.embeddings_file, "w") as f:
                json.dump(self.embeddings, f)
        except Exception as e:
            print(f"Failed to save embeddings: {e}")

    def load_model(self):
        if self.model is not None:
            return
        
        self.is_loading = True
        try:
            print("Loading CLIP model...")
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        finally:
            self.is_loading = False

    def index_images(self, image_paths):
        if self.is_indexing:
            return
            
        self.is_indexing = True
        self.index_progress["total"] = len(image_paths)
        self.index_progress["current"] = 0
        self.index_progress["status"] = "indexing"
        
        threading.Thread(target=self._index_loop, args=(image_paths,), daemon=True).start()

    def _index_loop(self, image_paths):
        try:
            self.load_model()
            import torch
            from PIL import Image

            # Filter out already indexed paths
            to_process = [p for p in image_paths if p not in self.embeddings and os.path.exists(p)]
            self.index_progress["total"] = len(to_process) # Reset total to what's actually needed
            
            # Save every N images
            batch_size = 10
            processed_in_batch = 0

            # Batch processing for speed
            batch_size = 32
            current_batch_paths = []
            current_batch_images = []
            
            print(f"Starting indexing of {len(to_process)} images in batches of {batch_size}...")

            for i, path in enumerate(to_process):
                self.index_progress["current"] += 1
                try:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in VIDEO_EXTS: continue # Skip videos as requested

                    image = Image.open(path)
                    # Force load to check for corruption early
                    image.load()
                    current_batch_paths.append(path)
                    current_batch_images.append(image)
                    
                except Exception as e:
                    print(f"Skipping corrupted file {path}: {e}")
                    continue

                # Process batch if full or last item
                if len(current_batch_images) >= batch_size or i == len(to_process) - 1:
                    if not current_batch_images:
                        continue
                        
                    try:
                        inputs = self.processor(images=current_batch_images, return_tensors="pt", padding=True).to(self.device)
                        
                        with torch.no_grad():
                            batch_features = self.model.get_image_features(**inputs)
                        
                        # Handle possibility of ModelOutput object (extract tensor)
                        if hasattr(batch_features, 'image_embeds'):
                            batch_features = batch_features.image_embeds
                        elif hasattr(batch_features, 'pooler_output'):
                            batch_features = batch_features.pooler_output

                        # Normalize and store
                        batch_features /= batch_features.norm(dim=-1, keepdim=True)
                        features_list = batch_features.cpu().tolist()
                        
                        for p, feat in zip(current_batch_paths, features_list):
                            self.embeddings[p] = feat
                        
                        if self.index_progress["current"] % 500 < batch_size:
                             self.save_embeddings()
                             print(f"Progress: {self.index_progress['current']} / {len(to_process) + self.index_progress['current'] - (i+1)} indexed (Saved)...")
                             
                    except Exception as e:
                        print(f"Batch processing failed: {e}. Falling back to single-image for this batch.")
                        # If a batch fails (e.g. one image is still weird), process one by one
                        for single_image, single_path in zip(current_batch_images, current_batch_paths):
                            try:
                                inputs = self.processor(images=single_image, return_tensors="pt").to(self.device)
                                with torch.no_grad():
                                    feat = self.model.get_image_features(**inputs)
                                    
                                if hasattr(feat, 'image_embeds'):
                                    feat = feat.image_embeds
                                elif hasattr(feat, 'pooler_output'):
                                    feat = feat.pooler_output

                                feat /= feat.norm(dim=-1, keepdim=True)
                                self.embeddings[single_path] = feat.cpu().tolist()[0]
                            except:
                                pass
                    
                    current_batch_paths = []
                    current_batch_images = []
            
            self.save_embeddings()
            print("Indexing complete.")
            
        except Exception as e:
            print(f"Indexing failed: {e}")
        finally:
            self.is_indexing = False
            self.index_progress["status"] = "idle"

    def search(self, query_text, cutoff=0.2, limit=100, inverse=False):
        self.load_model()
        import torch
        import numpy as np
        
        # Prompt Ensembling: CLIP performs significantly better when queries are framed as descriptions.
        prompts = [
            query_text,
            f"a photo of a {query_text}",
            f"a picture of a {query_text}",
            f"an image showing {query_text}",
            f"a high-quality photo of {query_text}"
        ]
        
        input_ids = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**input_ids)
            
        # Extract features and average them
        if hasattr(text_features, 'text_embeds'):
            text_features = text_features.text_embeds
        elif hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
            
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Average the prompts and normalize the resulting vector
        text_vec = text_features.mean(dim=0, keepdim=True)
        text_vec /= text_vec.norm(dim=-1, keepdim=True)
        text_vec = text_vec.cpu().numpy()[0]
        
        results = []
        
        # We process manually or using matrix math if we converted all to numpy. 
        # For simplicity and robust handling of dict updates, we iterate.
        # Optimization: build matrix on demand if slow.
        
        paths = list(self.embeddings.keys())
        if not paths:
            return []
            
        # Convert to matrix
        matrix = np.array([self.embeddings[p] for p in paths])
        scores = (matrix @ text_vec).flatten()
        
        # Filter and sort
        combined = []
        for i, score in enumerate(scores):
            if inverse:
                 if score < cutoff:
                    combined.append({"path": paths[i], "score": float(score)})
            else:
                if score > cutoff:
                    combined.append({"path": paths[i], "score": float(score)})
                
        # If inverse, sort ascending (lowest score likely matches "not query")
        # If normal, sort descending
        if inverse:
             combined.sort(key=lambda x: x["score"])
        else:
             combined.sort(key=lambda x: -x["score"])
             
        return combined[:limit]

semantic_search = SemanticSearch()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return Response(HTML_TEMPLATE, mimetype="text/html")


@app.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename):
    return send_from_directory(THUMB_DIR, filename)


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    global ROOT_DIR, scan_complete, duplicates, all_media_files
    if request.method == "POST":
        data = request.get_json()
        new_root = data.get("root_dir")
        if new_root:
            new_root = os.path.expanduser(new_root)
            # Add trailing slash if not present as requested by user
            if not new_root.endswith(os.sep):
                new_root += os.sep
                
            if os.path.isdir(new_root):
                ROOT_DIR = os.path.abspath(new_root)
                # Ensure it has trailing slash for consistency if requested
                if not ROOT_DIR.endswith(os.sep):
                    ROOT_DIR += os.sep
                
                # Save to settings
                settings = load_settings()
                settings["root_dir"] = ROOT_DIR
                save_settings(settings)
                
                # Reset state and restart scan
                duplicates = {}
                all_media_files = []
                scan_complete = False
                scan_stats["phase"] = "idle"
                scan_stats["phase_detail"] = "Restarting scan for new directory..."
                
                threading.Thread(target=run_scan, daemon=True).start()
                return jsonify({"success": True, "root_dir": ROOT_DIR})
            else:
                return jsonify({"success": False, "error": f"Invalid directory: {new_root}"}), 400
        return jsonify({"success": False, "error": "No directory provided"}), 400
    return jsonify({"root_dir": ROOT_DIR})


@app.route("/api/stats")
def api_stats():
    return jsonify(compute_stats())


@app.route("/api/groups")
def api_groups():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    sort = request.args.get("sort", "size_desc")

    groups_list = []
    for h, files in duplicates.items():
        groups_list.append({
            "hash": h,
            "file_size": files[0]["size"],
            "file_size_human": format_size(files[0]["size"]),
            "count": len(files),
            "files": [
                {
                    "path": fi["path"],
                    "size": fi["size"],
                    "size_human": format_size(fi["size"]),
                    "type": fi["type"],
                    "thumbnail": get_thumb_filename(fi["path"]),
                    "folder": os.path.dirname(fi["path"]),
                }
                for fi in files
            ],
        })

    if sort == "size_desc":
        groups_list.sort(key=lambda g: -g["file_size"])
    elif sort == "size_asc":
        groups_list.sort(key=lambda g: g["file_size"])

    total = len(groups_list)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page

    return jsonify({
        "groups": groups_list[start:end],
        "page": page,
        "per_page": per_page,
        "total_groups": total,
        "total_pages": total_pages,
    })


@app.route("/api/folders")
def api_folders():
    return jsonify(compute_folder_stats())


@app.route("/api/delete", methods=["POST"])
def api_delete():
    data = request.get_json(force=True)
    path = data.get("path")
    if not path:
        return jsonify({"success": False, "error": "No path provided"}), 400

    size = 0
    # Try to get the size before deletion
    try:
        size = os.path.getsize(path)
    except OSError:
        pass

    # Find kept versions before deleting
    kept_versions = []
    for h, files in duplicates.items():
        # Check if the file to be deleted is in this group
        if any(f["path"] == path for f in files):
            # Record others in the group
            for f in files:
                if f["path"] != path:
                    kept_versions.append(f["path"])
            break

    ok, err = move_to_trash(path)
    if not ok:
        return jsonify({"success": False, "error": err}), 400

    remove_file_from_duplicates(path)
    remove_file_from_index(path)

    # Record in history
    deletion_history["total_space_saved"] += size
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        deletion_history["total_videos_deleted"] = deletion_history.get("total_videos_deleted", 0) + 1
        deletion_history["space_saved_videos"] = deletion_history.get("space_saved_videos", 0) + size
    else:
        deletion_history["total_images_deleted"] = deletion_history.get("total_images_deleted", 0) + 1
        deletion_history["space_saved_images"] = deletion_history.get("space_saved_images", 0) + size
    deletion_history["deletions"].append({
        "path": path,
        "size": size,
        "size_human": format_size(size),
        "timestamp": datetime.now().isoformat(),
        "kept_versions": kept_versions
    })
    save_history(deletion_history)

    return jsonify({
        "success": True,
        "size_freed": size,
        "size_freed_human": format_size(size),
        "total_space_saved": deletion_history["total_space_saved"],
        "total_space_saved_human": format_size(deletion_history["total_space_saved"]),
    })


@app.route("/api/delete-batch", methods=["POST"])
def api_delete_batch():
    data = request.get_json(force=True)
    paths = data.get("paths", [])
    if not paths:
        return jsonify({"success": False, "error": "No paths provided"}), 400

    results = []
    total_freed = 0
    for path in paths:
        size = 0
        try:
            size = os.path.getsize(path)
        except OSError:
            pass

        ok, err = move_to_trash(path)
        if ok:
            # Find kept versions for this specific file
            kept_versions = []
            for h, files in duplicates.items():
                if any(f["path"] == path for f in files):
                     for f in files:
                        if f["path"] != path:
                            kept_versions.append(f["path"])
                     break

            remove_file_from_duplicates(path)
            remove_file_from_index(path)
            deletion_history["total_space_saved"] += size
            ext = os.path.splitext(path)[1].lower()
            if ext in VIDEO_EXTS:
                deletion_history["total_videos_deleted"] = deletion_history.get("total_videos_deleted", 0) + 1
                deletion_history["space_saved_videos"] = deletion_history.get("space_saved_videos", 0) + size
            else:
                deletion_history["total_images_deleted"] = deletion_history.get("total_images_deleted", 0) + 1
                deletion_history["space_saved_images"] = deletion_history.get("space_saved_images", 0) + size
            deletion_history["deletions"].append({
                "path": path,
                "size": size,
                "size_human": format_size(size),
                "timestamp": datetime.now().isoformat(),
                "kept_versions": kept_versions
            })
            total_freed += size
            results.append({"path": path, "success": True})
        else:
            results.append({"path": path, "success": False, "error": err})

    save_history(deletion_history)

    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded

    return jsonify({
        "success": failed == 0,
        "results": results,
        "succeeded": succeeded,
        "failed": failed,
        "total_freed": total_freed,
        "total_freed_human": format_size(total_freed),
        "total_space_saved": deletion_history["total_space_saved"],
        "total_space_saved_human": format_size(deletion_history["total_space_saved"]),
    })


@app.route("/api/history")
def api_history():
    updated = 0
    if request.args.get("enrich") == "true":
        updated = enrich_history_from_index()
    return jsonify({
        "total_space_saved": deletion_history["total_space_saved"],
        "deletions": deletion_history["deletions"],
        "enriched_count": updated
    })


@app.route("/api/folder-rules", methods=["POST"])
def api_folder_rules():
    """Preview or execute folder-priority deletion across ALL duplicate groups.

    Body: {"keep_folders": [...], "dry_run": true/false}

    Logic per group:
      - If any file's directory starts with a keep_folder prefix -> keep those,
        delete the rest.
      - If NO file is in a keep folder -> skip the group (nothing deleted).
      - If ALL files are in keep folders -> skip (nothing to delete).
    """
    data = request.get_json(force=True)
    keep_folders = data.get("keep_folders", [])
    dry_run = data.get("dry_run", True)

    if not keep_folders:
        return jsonify({"success": False, "error": "No keep_folders provided"}), 400

    # Normalise: ensure trailing sep for prefix matching
    normalised = []
    for kf in keep_folders:
        kf = os.path.realpath(kf)
        if not kf.endswith(os.sep):
            kf += os.sep
        normalised.append(kf)

    def _in_keep(path):
        folder = os.path.dirname(os.path.realpath(path))
        if not folder.endswith(os.sep):
            folder += os.sep
        return any(folder.startswith(kf) or folder == kf.rstrip(os.sep) for kf in normalised)

    to_delete = []  # list of {"path", "size", "type", "group_hash"}
    groups_affected = 0

    for h, files in list(duplicates.items()):
        # Split files into those in 'keep' folders and those not
        kept = [f for f in files if _in_keep(f["path"])]
        others = [f for f in files if not _in_keep(f["path"])]
        
        # If we have files in keep folders, delete all duplicates in other (un-kept) folders
        if kept and others:
            groups_affected += 1
            for f in others:
                to_delete.append({"path": f["path"], "size": f["size"], "type": f.get("type", "photo"), "group_hash": h})
        # Note: We intentionally skip groups that are entirely inside keep folders
        # OR entirely outside keep folders, to avoid accidental mass deletion.

    total_size = sum(f["size"] for f in to_delete)

    if dry_run:
        # Group files by folder for the visual preview
        by_folder = defaultdict(list)
        for f in to_delete:
            folder = os.path.dirname(f["path"])
            by_folder[folder].append({
                "path": f["path"],
                "size": f["size"],
                "size_human": format_size(f["size"]),
                "type": f["type"],
                "thumbnail": get_thumb_filename(f["path"]),
            })

        return jsonify({
            "dry_run": True,
            "files_to_delete": len(to_delete),
            "total_size": total_size,
            "total_size_human": format_size(total_size),
            "groups_affected": groups_affected,
            "by_folder": {folder: files for folder, files in by_folder.items()},
        })

    # Execute deletions
    succeeded = 0
    failed = 0
    freed = 0
    errors = []
    for f in to_delete:
        size = f["size"]
        ok, err = move_to_trash(f["path"])
        if ok:
            remove_file_from_duplicates(f["path"])
            deletion_history["total_space_saved"] += size
            ext = os.path.splitext(f["path"])[1].lower()
            if ext in VIDEO_EXTS:
                deletion_history["total_videos_deleted"] = deletion_history.get("total_videos_deleted", 0) + 1
                deletion_history["space_saved_videos"] = deletion_history.get("space_saved_videos", 0) + size
            else:
                deletion_history["total_images_deleted"] = deletion_history.get("total_images_deleted", 0) + 1
                deletion_history["space_saved_images"] = deletion_history.get("space_saved_images", 0) + size
            deletion_history["deletions"].append({
                "path": f["path"],
                "size": size,
                "size_human": format_size(size),
                "timestamp": datetime.now().isoformat(),
            })
            freed += size
            succeeded += 1
        else:
            failed += 1
            if len(errors) < 20:
                errors.append({"path": f["path"], "error": err})

    save_history(deletion_history)

    return jsonify({
        "dry_run": False,
        "succeeded": succeeded,
        "failed": failed,
        "freed": freed,
        "freed_human": format_size(freed),
        "groups_affected": groups_affected,
        "total_space_saved": deletion_history["total_space_saved"],
        "total_space_saved_human": format_size(deletion_history["total_space_saved"]),
        "errors": errors,
    })


@app.route("/api/empty-folders")
def api_empty_folders():
    return jsonify(get_empty_folders())


@app.route("/api/delete-empty-folders", methods=["POST"])
def api_delete_empty_folders():
    data = request.get_json(force=True)
    paths = data.get("paths", [])
    results = []
    for path in paths:
        if not is_within_root(path):
            results.append({"path": path, "success": False, "error": "Outside root"})
            continue
        try:
            if os.path.isdir(path) and is_effectively_empty(path):
                # Delete .DS_Store if it exists
                ds_store = os.path.join(path, ".DS_Store")
                if os.path.exists(ds_store):
                    os.unlink(ds_store)
                os.rmdir(path)
                results.append({"path": path, "success": True})
            else:
                results.append({"path": path, "success": False, "error": "Not empty or not a directory"})
        except Exception as e:
            results.append({"path": path, "success": False, "error": str(e)})
    return jsonify({
        "success": all(r["success"] for r in results),
        "results": results
    })


@app.route("/api/small-files")
def api_small_files():
    max_size_mb = request.args.get("max_size_mb", 0.5, type=float)
    min_size_mb = request.args.get("min_size_mb", 0.0, type=float)
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 200, type=int)
    
    max_size_bytes = max_size_mb * 1024 * 1024
    min_size_bytes = min_size_mb * 1024 * 1024
    
    # Filter all_media_files
    matches = [f for f in all_media_files if min_size_bytes <= f["size"] < max_size_bytes]
    
    # Sort by size ascending (smallest first)
    matches.sort(key=lambda x: x["size"])
    
    total = len(matches)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    
    results = []
    for f in matches[start:end]:
        make_thumbnail(f["path"])
        results.append({
            "path": f["path"],
            "size": f["size"],
            "size_human": format_size(f["size"]),
            "type": f["type"],
            "thumbnail": get_thumb_filename(f["path"])
        })

    return jsonify({
        "files": results,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    })


@app.route("/api/semantic-search")
def api_semantic_search():
    query = request.args.get("q", "")
    cutoff = request.args.get("threshold", 0.2, type=float)
    limit = request.args.get("limit", 100, type=int)
    inverse = request.args.get("inverse", "false") == "true"
    
    # If inverse, we might want a lower/different default cutoff if not specified? 
    # But usually < 0.2 is fine for "likely not match". 
    # Actually, for "non-human", we want score < 0.15 or so.
    # Let frontend control cutoff.
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
        
    try:
        results = semantic_search.search(query, cutoff=cutoff, limit=limit, inverse=inverse)
        
        # Enrich results with thumbnails and human sizes
        enriched = []
        for r in results:
            path = r["path"]
            try:
                size = os.path.getsize(path)
                make_thumbnail(path)
                enriched.append({
                    "path": path,
                    "score": r["score"],
                    "size": size,
                    "size_human": format_size(size),
                    "thumbnail": get_thumb_filename(path),
                    "type": "video" if os.path.splitext(path)[1].lower() in VIDEO_EXTS else "photo"
                })
            except OSError:
                continue
                
        return jsonify({"results": enriched})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/semantic-status")
def api_semantic_status():
    return jsonify({
        "is_loading": semantic_search.is_loading,
        "is_indexing": semantic_search.is_indexing,
        "progress": semantic_search.index_progress,
        "indexed_count": len(semantic_search.embeddings)
    })


@app.route("/api/semantic-index", methods=["POST"])
def api_semantic_index():
    # Use the globally found all_media_files
    # Note: These are populated after a scan run. If scan hasn't run, we might need to trigger it or scan partially.
    # For now, we assume user ran a scan or we just walk the dir again if needed.
    
    targets = [f["path"] for f in all_media_files if os.path.splitext(f["path"])[1].lower() in PHOTO_EXTS]
    if not targets:
        # Fallback if scan hasn't run yet: just walk quickly? 
        # Or better, just tell user to wait for scan?
        # Let's try to just collect them quickly if empty.
        raw_files = []
        for dirpath, _, filenames in os.walk(ROOT_DIR):
              for name in filenames:
                  ext = os.path.splitext(name)[1].lower()
                  if ext in PHOTO_EXTS:
                      raw_files.append(os.path.join(dirpath, name))
        targets = raw_files
        
    semantic_search.index_images(targets)
    return jsonify({"success": True, "message": "Indexing started"})


@app.route("/api/all-files")
def api_all_files():
    global all_media_files
    if not all_media_files:
        all_files_path = os.path.join(DATA_DIR, "all_files_index.json")
        if os.path.exists(all_files_path):
            try:
                with open(all_files_path, 'r') as f:
                    all_media_files = json.load(f)
            except Exception:
                pass
                
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 200, type=int)
    sort = request.args.get("sort", "size_desc")
    
    matches = list(all_media_files)
    
    if sort == "size_desc":
        matches.sort(key=lambda x: -x["size"])
    else:
        matches.sort(key=lambda x: x["size"])
    
    total = len(matches)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    
    results = []
    for f in matches[start:end]:
        make_thumbnail(f["path"])
        results.append({
            "path": f["path"],
            "size": f["size"],
            "size_human": format_size(f["size"]),
            "type": f["type"],
            "thumbnail": get_thumb_filename(f["path"])
        })

    return jsonify({
        "files": results,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    })


@app.route("/api/view-original")
def api_view_original():
    path = request.args.get("path")
    if not path or not is_within_root(path):
        return "Not found", 404
    return send_from_directory(os.path.dirname(path), os.path.basename(path))


# ---------------------------------------------------------------------------
# Embedded HTML SPA
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Photo Clean-Up</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ========================================
   Modern Minimal Design System
   ======================================== */

:root {
    /* Colors */
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-tertiary: #f1f5f9;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --border-color: #e2e8f0;
    --border-light: #f1f5f9;
    
    /* Accent Colors */
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --accent-light: #eef2ff;
    --success: #10b981;
    --success-light: #ecfdf5;
    --success-border: #a7f3d0;
    --danger: #f43f5e;
    --danger-hover: #e11d48;
    --danger-light: #fff1f2;
    --danger-border: #fecdd3;
    --warning: #f59e0b;
    --warning-light: #fffbeb;
    
    /* Spacing */
    --space-xs: 4px;
    --space-sm: 8px;
    --space-md: 16px;
    --space-lg: 24px;
    --space-xl: 32px;
    
    /* Border Radius */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 20px;
    --radius-full: 9999px;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.06);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.08);
    --shadow-xl: 0 12px 40px rgba(0,0,0,0.12);
    
    /* Transitions */
    --transition-fast: 0.15s ease;
    --transition-normal: 0.2s ease;
    --transition-slow: 0.3s ease;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    padding: var(--space-lg);
    min-height: 100vh;
    line-height: 1.5;
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
}

h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
h1 { font-size: 1.75rem; color: var(--text-primary); margin-bottom: var(--space-lg); }
h2 { font-size: 1.25rem; color: var(--text-primary); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ========================================
   Stats Cards - Clean Flat Design
   ======================================== */
.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: var(--space-md);
    margin-bottom: var(--space-lg);
}

.stat-box {
    background: var(--bg-secondary);
    padding: var(--space-lg);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    transition: all var(--transition-normal);
}

.stat-box:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.stat-box h3 {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: var(--space-sm);
}

.stat-box .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.stat-box.green .value { color: var(--success); }
.stat-box.purple .value { color: var(--accent); }

/* ========================================
   Modern Pill Tabs
   ======================================== */
.tabs {
    display: flex;
    gap: var(--space-sm);
    margin-bottom: var(--space-lg);
    padding: var(--space-xs);
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    flex-wrap: wrap;
}

.tab {
    padding: var(--space-sm) var(--space-md);
    background: transparent;
    border: none;
    cursor: pointer;
    border-radius: var(--radius-md);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    transition: all var(--transition-fast);
    white-space: nowrap;
}

.tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.tab.active {
    background: var(--accent);
    color: white;
    font-weight: 600;
}

.tab.smart-tab {
    background: linear-gradient(135deg, #a78bfa 0%, #6366f1 100%);
    color: white;
}

.tab.smart-tab:not(.active) {
    background: var(--accent-light);
    color: var(--accent);
}

.tab-content {
    display: none;
    background: var(--bg-secondary);
    padding: var(--space-lg);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
}

.tab-content.active { display: block; }

/* ========================================
   Action Bar
   ======================================== */
.action-bar {
    position: sticky;
    top: var(--space-md);
    z-index: 100;
    background: var(--bg-secondary);
    padding: var(--space-md) var(--space-lg);
    border-radius: var(--radius-lg);
    margin-bottom: var(--space-lg);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-md);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-md);
}

.action-bar .summary {
    font-size: 13px;
    color: var(--text-secondary);
}

.action-bar .summary span { 
    font-weight: 700; 
    color: var(--accent);
}

.action-bar .summary strong {
    color: var(--danger);
}

.action-bar .controls {
    display: flex;
    gap: var(--space-sm);
    align-items: center;
    flex-wrap: wrap;
}

/* ========================================
   Modern Buttons
   ======================================== */
.btn {
    padding: var(--space-sm) var(--space-md);
    border: none;
    border-radius: var(--radius-full);
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all var(--transition-fast);
    display: inline-flex;
    align-items: center;
    gap: var(--space-xs);
    box-shadow: var(--shadow-sm);
}

.btn:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.btn:active { transform: translateY(0); }

.btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none !important;
    box-shadow: none !important;
}

.btn-primary {
    background: var(--accent);
    color: white;
}
.btn-primary:hover { background: var(--accent-hover); }

.btn-danger {
    background: var(--danger);
    color: white;
}
.btn-danger:hover { background: #dc2626; }

.btn-success, .btn-keep {
    background: var(--success);
    color: white;
}
.btn-success:hover, .btn-keep:hover { background: #059669; }

.btn-delete {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
}
.btn-delete:hover {
    background: var(--accent-light);
    color: var(--accent);
    border-color: var(--accent);
}

.delete-mark .btn-delete {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
}

.btn-secondary {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
    box-shadow: none;
}
.btn-secondary:hover {
    background: var(--border-color);
    color: var(--text-primary);
}

.btn-lg {
    padding: 10px 20px;
    font-size: 14px;
}

select.sort-select {
    padding: var(--space-sm) var(--space-md);
    font-size: 13px;
    border-radius: var(--radius-full);
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    color: var(--text-primary);
    cursor: pointer;
    outline: none;
    transition: all var(--transition-fast);
}

select.sort-select:hover { border-color: var(--accent); }
select.sort-select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-light); }

/* ========================================
   Groups & Cards
   ======================================== */
.group {
    margin-bottom: var(--space-md);
    background: var(--bg-secondary);
    padding: var(--space-lg);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    transition: all var(--transition-normal);
}

.group:hover { box-shadow: var(--shadow-sm); }

.group-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-sm);
    margin: calc(-1 * var(--space-sm));
    margin-bottom: var(--space-md);
    border-radius: var(--radius-md);
    cursor: pointer;
    user-select: none;
    transition: background var(--transition-fast);
}

.group-header:hover { background: var(--bg-tertiary); }

.group-header h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
}

.group-toggle-icon {
    display: inline-block;
    margin-right: var(--space-sm);
    transition: transform var(--transition-fast);
    color: var(--text-muted);
}

.group-body { margin-top: var(--space-md); }
.group.collapsed .group-toggle-icon { transform: rotate(-90deg); }

.images {
    display: grid;
    grid-template-columns: repeat(auto-fill, 120px);
    grid-auto-rows: 240px;
    gap: var(--space-md);
    justify-content: start;
}

/* ========================================
   Image Box - Modern Card Style
   ======================================== */
.imgbox {
    background: var(--bg-secondary);
    border: 2px solid var(--border-color);
    padding: 8px;
    border-radius: var(--radius-md);
    transition: all var(--transition-normal);
    position: relative;
    overflow: hidden;
    width: 120px;
    height: 240px;
    display: flex;
    flex-direction: column;
}

.imgbox:hover {
    border-color: var(--accent);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.imgbox.keep {
    border-color: var(--success);
    background: var(--success-light);
}

.imgbox.delete-mark {
    border-color: var(--accent);
    background: var(--accent-light);
    transform: scale(0.98);
}

/* Remove old indicator */
.imgbox.delete-mark::before { display: none; }

@keyframes popIn {
    from { transform: scale(0); }
    to { transform: scale(1); }
}

.imgbox .thumb-container {
    position: relative;
    aspect-ratio: 1/1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    overflow: hidden;
    cursor: pointer;
}

.imgbox img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: var(--radius-sm);
    transition: transform var(--transition-normal);
}

.imgbox:hover img { transform: scale(1.02); }

.imgbox .video-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 20px;
    color: white;
    text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    pointer-events: none;
    background: rgba(0,0,0,0.3);
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.imgbox .path {
    font-size: 10px;
    margin-top: var(--space-sm);
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.4;
    font-family: 'SF Mono', Monaco, monospace;
    pointer-events: none;
}

.imgbox .meta {
    font-size: 10px;
    color: var(--text-secondary);
    margin-top: 2px;
    font-weight: 500;
}

.type-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: var(--radius-full);
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-left: var(--space-xs);
}

.type-badge.video { background: #a78bfa; color: white; }
.type-badge.photo { background: #60a5fa; color: white; }

.imgbox .actions {
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: opacity var(--transition-fast);
}

.selection-mode .imgbox .actions {
    opacity: 0;
    pointer-events: none;
}

.imgbox .actions .btn {
    width: 100%;
    padding: 6px;
    font-size: 10px;
    justify-content: center;
    border-radius: var(--radius-sm);
    box-shadow: none;
}

.imgbox .actions .btn-danger {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid transparent;
}

.imgbox .actions .btn-danger:hover {
    background: var(--danger-light);
    color: var(--danger);
    border-color: var(--danger-border);
}

/* Choice Indicator */
.imgbox::after {
    content: "";
    position: absolute;
    top: 8px;
    left: 8px;
    width: 18px;
    height: 18px;
    border: 2px solid white;
    background: rgba(255,255,255,0.2);
    border-radius: 4px;
    opacity: 0;
    transition: all var(--transition-fast);
    z-index: 10;
    backdrop-filter: blur(2px);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.1);
}

.selection-mode .imgbox:hover::after,
.imgbox.delete-mark::after {
    opacity: 1;
}

.imgbox.delete-mark::after {
    background: var(--accent);
    border-color: var(--accent);
    content: "✓";
    color: white;
    font-size: 12px;
    font-weight: 800;
}

.selection-mode .imgbox {
    cursor: pointer;
    user-select: none;
}

/* Kept Info in History */
.kept-info {
    margin-top: var(--space-xs);
    padding: var(--space-xs) var(--space-sm);
    background: var(--success-light);
    border-radius: var(--radius-sm);
    font-size: 11px;
    color: #047857;
    border: 1px solid var(--success-border);
    display: inline-block;
}

.kept-info strong { font-weight: 600; }

/* ========================================
   Pagination
   ======================================== */
.pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: var(--space-md);
    margin: var(--space-lg) 0;
    padding: var(--space-md);
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
}

.pagination button {
    padding: var(--space-sm) var(--space-lg);
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    cursor: pointer;
    border-radius: var(--radius-full);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    transition: all var(--transition-fast);
}

.pagination button:hover:not(:disabled) {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
}

.pagination button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.page-info {
    font-size: 13px;
    color: var(--text-secondary);
}

.page-info span { font-weight: 600; color: var(--text-primary); }

/* ========================================
   Folder List
   ======================================== */
.folder-list {
    max-height: 500px;
    overflow-y: auto;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
}

.folder-item {
    padding: var(--space-md);
    border-bottom: 1px solid var(--border-light);
    display: flex;
    align-items: center;
    gap: var(--space-md);
    transition: background var(--transition-fast);
}

.folder-item:last-child { border-bottom: none; }
.folder-item:hover { background: var(--bg-tertiary); }

.folder-item input[type="checkbox"] {
    width: 18px;
    height: 18px;
    cursor: pointer;
    flex-shrink: 0;
    accent-color: var(--accent);
}

.folder-item.kept {
    background: var(--success-light);
    border-left: 4px solid var(--success);
}

.folder-item.to-delete {
    background: var(--danger-light);
    border-left: 4px solid var(--danger);
}

.folder-path {
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 12px;
    word-break: break-all;
    flex: 1;
    color: var(--text-secondary);
}

.folder-count {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    padding: 4px 12px;
    border-radius: var(--radius-full);
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
}

.folder-count.delete-badge { 
    background: var(--danger); 
    color: white; 
    box-shadow: 0 2px 4px rgba(244, 63, 94, 0.2);
}
.folder-count.keep-badge { 
    background: var(--success); 
    color: white; 
    box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
}

.folder-actions {
    display: flex;
    gap: var(--space-md);
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: var(--space-md);
    padding: var(--space-md);
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
}

.folder-actions .info {
    font-size: 13px;
    color: var(--text-secondary);
    flex: 1;
}

.preview-box {
    margin: var(--space-md) 0;
    padding: var(--space-md);
    background: var(--warning-light);
    border: 1px solid #fcd34d;
    border-radius: var(--radius-md);
    font-size: 13px;
}

.preview-box strong { color: var(--warning); }

/* ========================================
   History Section
   ======================================== */

/* Statistics Grid */
.history-stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: var(--space-md);
    margin-bottom: var(--space-lg);
}

.history-stat-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: var(--space-lg);
    display: flex;
    align-items: center;
    gap: var(--space-md);
    transition: all var(--transition-normal);
    position: relative;
    overflow: hidden;
}

.history-stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    opacity: 0;
    transition: opacity var(--transition-normal);
}

.history-stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    border-color: var(--primary-light);
}

.history-stat-card:hover::before {
    opacity: 1;
}

.stat-icon {
    width: 56px;
    height: 56px;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.stat-icon svg {
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
}

.stat-content {
    flex: 1;
    min-width: 0;
}

.stat-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 4px;
}

.stat-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.stat-sublabel {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
}

/* Search and Filter Controls */
.history-controls {
    display: flex;
    gap: var(--space-md);
    margin-bottom: var(--space-lg);
    flex-wrap: wrap;
    align-items: center;
}

.history-search-box {
    flex: 1;
    min-width: 250px;
    position: relative;
    display: flex;
    align-items: center;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 0 var(--space-md);
    transition: all var(--transition-fast);
}

.history-search-box:focus-within {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.1);
}

.history-search-box svg {
    color: var(--text-muted);
    flex-shrink: 0;
    margin-right: var(--space-sm);
}

.history-search-box input {
    flex: 1;
    border: none;
    background: transparent;
    padding: 10px 0;
    font-size: 14px;
    color: var(--text-primary);
    outline: none;
}

.history-search-box input::placeholder {
    color: var(--text-muted);
}

.history-filter-group {
    display: flex;
    gap: var(--space-sm);
}

.history-filter-select {
    padding: 10px 14px;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
    outline: none;
}

.history-filter-select:hover {
    border-color: var(--primary-light);
}

.history-filter-select:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.1);
}

/* History Folder Sections */
.history-folder-section { 
    margin-bottom: var(--space-md);
    border-radius: var(--radius-lg);
    overflow: hidden;
    border: 1px solid var(--border-color);
    transition: all var(--transition-fast);
}

.history-folder-section:hover {
    border-color: var(--primary-light);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.history-folder-header {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-md) var(--space-lg);
    background: var(--bg-tertiary);
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    cursor: pointer;
    user-select: none;
    transition: background var(--transition-fast);
    border-bottom: 1px solid var(--border-light);
}

.history-folder-header:hover { 
    background: var(--bg-secondary);
}

.history-folder-header .hist-folder-toggle {
    font-size: 14px;
    color: var(--text-muted);
    flex-shrink: 0;
    transition: transform var(--transition-fast);
    display: inline-block;
}

.history-folder-header .hist-folder-toggle.open { 
    transform: rotate(90deg);
}

.history-folder-header .hist-folder-label { 
    flex: 1;
    word-break: break-all;
    display: flex;
    align-items: center;
    gap: var(--space-sm);
}

.history-folder-body {
    background: var(--bg-secondary);
    max-height: 600px;
    overflow-y: auto;
}

.history-folder-body.collapsed { 
    display: none;
}

/* History Items */
.history-item {
    padding: var(--space-md) var(--space-lg);
    border-bottom: 1px solid var(--border-light);
    display: flex;
    align-items: flex-start;
    gap: var(--space-md);
    font-size: 13px;
    transition: background var(--transition-fast);
}

.history-item:hover {
    background: var(--bg-tertiary);
}

.history-item:last-child { 
    border-bottom: none;
}

.history-item-icon {
    width: 40px;
    height: 40px;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 18px;
}

.history-item-icon.photo {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
}

.history-item-icon.video {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
}

.history-item-content {
    flex: 1;
    min-width: 0;
}

.history-item .hist-path {
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 13px;
    font-weight: 500;
    word-break: break-all;
    color: var(--text-primary);
    margin-bottom: 4px;
    display: block;
}

.history-item .kept-info {
    margin-top: var(--space-sm);
    padding: var(--space-sm);
    background: var(--bg-tertiary);
    border-left: 3px solid var(--success);
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--text-secondary);
}

.history-item .kept-info strong {
    color: var(--success);
    font-weight: 600;
}

.history-item .hist-meta {
    text-align: right;
    white-space: nowrap;
    color: var(--text-muted);
    font-size: 12px;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
}

.history-item .hist-size {
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 13px;
}

.history-item .hist-timestamp {
    font-size: 11px;
    opacity: 0.8;
}

/* ========================================
   Toast Notifications
   ======================================== */
.toast-container {
    position: fixed;
    top: var(--space-lg);
    right: var(--space-lg);
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
}

.toast {
    padding: var(--space-md) var(--space-lg);
    border-radius: var(--radius-lg);
    color: white;
    font-size: 13px;
    font-weight: 500;
    box-shadow: var(--shadow-xl);
    animation: toastSlide 0.3s ease;
    max-width: 400px;
    backdrop-filter: blur(8px);
}

.toast.success { background: var(--success); }
.toast.error { background: var(--danger); }
.toast.info { background: var(--accent); }

@keyframes toastSlide {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}

/* ========================================
   Scan Overlay
   ======================================== */
.scan-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(4px);
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
}

.scan-box {
    background: var(--bg-secondary);
    padding: var(--space-xl) 48px;
    border-radius: var(--radius-xl);
    text-align: center;
    max-width: 420px;
    box-shadow: var(--shadow-xl);
}

.scan-box h2 {
    margin-bottom: var(--space-sm);
    color: var(--text-primary);
}

.scan-box .phase {
    color: var(--text-secondary);
    font-size: 14px;
    margin-top: var(--space-sm);
}

.spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: var(--space-md) auto;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* ========================================
   Empty State
   ======================================== */
.empty {
    text-align: center;
    padding: 60px var(--space-lg);
    color: var(--text-muted);
}

.empty .icon {
    font-size: 48px;
    margin-bottom: var(--space-md);
    opacity: 0.5;
}

.empty p { margin-top: var(--space-sm); }

/* ========================================
   Preview Section
   ======================================== */
.preview-summary-bar {
    padding: var(--space-md) var(--space-lg);
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    margin-bottom: var(--space-md);
    border: 1px solid var(--border-color);
    font-size: 14px;
    color: var(--text-secondary);
}

.preview-summary-bar strong { color: var(--text-primary); }

.preview-folder-section {
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    margin-bottom: var(--space-md);
    border: 1px solid var(--border-color);
    overflow: hidden;
}

.preview-folder-header {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    padding: var(--space-md);
    cursor: pointer;
    user-select: none;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-color);
    transition: background var(--transition-fast);
}

.preview-folder-header:hover { background: var(--border-color); }

.preview-folder-header .folder-toggle {
    font-size: 12px;
    color: var(--text-muted);
    transition: transform var(--transition-fast);
    flex-shrink: 0;
}

.preview-folder-header .folder-toggle.open { transform: rotate(90deg); }

.preview-folder-header .folder-name {
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 12px;
    word-break: break-all;
    flex: 1;
    color: var(--text-primary);
}

.preview-folder-header .folder-badge {
    background: var(--danger);
    color: white;
    padding: 4px 12px;
    border-radius: var(--radius-full);
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
}

.preview-folder-header .folder-check-toggle {
    padding: 6px 12px;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-full);
    font-size: 11px;
    cursor: pointer;
    background: var(--bg-secondary);
    color: var(--text-secondary);
    white-space: nowrap;
    transition: all var(--transition-fast);
}

.preview-folder-header .folder-check-toggle:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.preview-folder-body { padding: var(--space-md); }
.preview-folder-body.collapsed { display: none; }

.preview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, 120px);
    grid-auto-rows: 180px;
    gap: var(--space-md);
    justify-content: start;
}

.preview-card {
    border: 1px solid var(--border-color);
    padding: 8px;
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
    transition: all var(--transition-normal);
    position: relative;
    overflow: hidden;
    width: 120px;
    height: 180px;
    display: flex;
    flex-direction: column;
}

.preview-card:not(.unchecked) {
    border-color: var(--danger);
    background: var(--danger-light);
    box-shadow: 0 0 0 1px var(--danger);
}

.preview-card.unchecked {
    opacity: 0.6;
    filter: grayscale(0.5);
}

.preview-card .pc-checkbox {
    position: absolute;
    top: 8px;
    left: 8px;
    z-index: 5;
    width: 16px;
    height: 16px;
    cursor: pointer;
    accent-color: var(--danger);
    filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));
}

.preview-card .pc-thumb {
    width: 100%;
    aspect-ratio: 1/1;
    object-fit: cover;
    border-radius: var(--radius-sm);
    display: block;
    background: var(--bg-tertiary);
}

.preview-card .pc-name {
    font-size: 10px;
    margin-top: var(--space-xs);
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
    font-family: 'SF Mono', Monaco, monospace;
}

.preview-card .pc-meta {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
    font-weight: 500;
}

.preview-card .video-badge {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 20px;
    color: white;
    text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    pointer-events: none;
}

.preview-bottom-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 200;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-color);
    padding: var(--space-md) var(--space-xl);
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.08);
}

.preview-bottom-bar .pbb-info {
    font-size: 14px;
    color: var(--text-secondary);
}

.preview-bottom-bar .pbb-info strong { color: var(--text-primary); }
.preview-bottom-bar .pbb-actions { display: flex; gap: var(--space-md); }

/* ========================================
   Modal
   ======================================== */
.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(8px);
    z-index: 11000;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}

.modal.active { display: flex; }

.modal-close {
    position: absolute;
    top: var(--space-lg);
    right: var(--space-xl);
    color: white;
    font-size: 36px;
    cursor: pointer;
    user-select: none;
    opacity: 0.8;
    transition: opacity var(--transition-fast);
}

.modal-close:hover { opacity: 1; }

.modal-content {
    max-width: 95%;
    max-height: 80%;
    object-fit: contain;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-xl);
}

.modal-caption {
    color: rgba(255,255,255,0.8);
    margin-top: var(--space-lg);
    font-size: 13px;
    text-align: center;
    word-break: break-all;
    max-width: 80%;
    font-family: 'SF Mono', Monaco, monospace;
}

.modal-actions {
    margin-top: var(--space-md);
    display: flex;
    gap: var(--space-md);
}

/* ========================================
   Smart Search Specific
   ======================================== */
#smart-search input[type="text"] {
    flex: 1;
    padding: var(--space-md);
    border: 2px solid var(--border-color);
    border-radius: var(--radius-lg);
    font-size: 16px;
    font-family: inherit;
    outline: none;
    transition: all var(--transition-fast);
}

#smart-search input[type="text"]:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px var(--accent-light);
}

#smart-status {
    margin-top: var(--space-md);
    font-size: 13px;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    padding: var(--space-sm) var(--space-md);
    border-radius: var(--radius-md);
    display: inline-flex;
    align-items: center;
    gap: var(--space-sm);
}

#smart-progress-container {
    max-width: 400px;
    margin: var(--space-md) auto;
}

#smart-progress-container > div:first-child {
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: var(--radius-full);
    overflow: hidden;
}

#smart-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), #a78bfa);
    border-radius: var(--radius-full);
    transition: width var(--transition-normal);
}

/* ========================================
   Responsive Adjustments
   ======================================== */
/* ========================================
   Cleanup Sub-tabs
   ======================================== */
.cleanup-sub-tabs {
    display: flex;
    gap: var(--space-xs);
    padding: var(--space-xs);
    background: var(--bg-tertiary);
    border-radius: var(--radius-md);
    width: fit-content;
    margin-bottom: var(--space-lg);
}

.cleanup-sub-tab {
    padding: var(--space-sm) var(--space-md);
    background: transparent;
    border: none;
    cursor: pointer;
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    transition: all var(--transition-fast);
}

.cleanup-sub-tab:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
}

.cleanup-sub-tab.active {
    background: var(--bg-secondary);
    color: var(--text-primary);
    box-shadow: var(--shadow-sm);
}

.cleanup-view { display: none; }
.cleanup-view.active { display: block; }

/* ========================================
   Responsive Adjustments
   ======================================== */
@media (max-width: 768px) {
    body { padding: var(--space-md); }
    
    .stats { grid-template-columns: repeat(2, 1fr); }
    
    .tabs { overflow-x: auto; flex-wrap: nowrap; }
    
    .action-bar {
        flex-direction: column;
        align-items: stretch;
    }
    
    .action-bar .controls {
        justify-content: center;
    }
    
    .images {
        grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    }
}
</style>
</head>
<body>

<div class="toast-container" id="toast-container"></div>
<div class="scan-overlay" id="scan-overlay" style="display:none">
  <div class="scan-box">
    <div class="spinner"></div>
    <h2>Scanning files...</h2>
    <div class="phase" id="scan-phase">Starting...</div>
  </div>
</div>

<!-- Full Resolution Modal -->
<div id="full-res-modal" class="modal">
  <span class="modal-close">&times;</span>
  <img id="modal-img" class="modal-content" src="" alt="">
  <video id="modal-video" class="modal-content" controls style="display:none"></video>
  <div id="modal-caption" class="modal-caption"></div>
  <div class="modal-actions">
    <a id="modal-download" class="btn btn-primary" href="" target="_blank">Download / Open Original</a>
  </div>
</div>

<header style="display:flex; justify-content:space-between; align-items:center; margin-bottom: var(--space-lg); flex-wrap: wrap; gap: var(--space-md);">
  <h1>Photo Clean-Up</h1>
  <div id="path-selector" style="display:flex; align-items:center; gap:var(--space-md);">
    <div style="display:flex; align-items:center; gap:var(--space-sm); background:var(--bg-secondary); padding:var(--space-sm) var(--space-md); border-radius:var(--radius-lg); border:1px solid var(--border-color); box-shadow:var(--shadow-sm);">
      <span style="font-size:12px; color:var(--text-secondary); font-weight:500;">Scanning:</span>
      <span id="current-root-path" style="font-family:'SF Mono', Monaco, monospace; font-size:12px; color:var(--text-primary); max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="Loading...">...</span>
      <button class="btn btn-secondary" id="btn-change-root" style="padding: 4px 10px; font-size: 11px;">Change</button>
    </div>
    <button class="btn btn-secondary" id="btn-toggle-selection-mode" style="padding: 10px 16px; font-size: 13px; font-weight: 600; min-width: 140px; justify-content: center;">
      <span id="selection-mode-icon">🔳</span> Bulk Edit
    </button>
  </div>
</header>

<div class="stats" id="stats-bar">
  <div class="stat-box"><h3>Duplicates</h3><div class="value" id="stat-groups">--</div></div>
  <div class="stat-box"><h3>Wasted Space</h3><div class="value" id="stat-wasted">--</div></div>
  <div class="stat-box green"><h3>Total Saved</h3><div class="value" id="stat-saved">--</div></div>
  <div class="stat-box"><h3>Images Deleted</h3><div class="value" id="stat-images-deleted">--</div></div>
  <div class="stat-box"><h3>Videos Deleted</h3><div class="value" id="stat-videos-deleted">--</div></div>
  <div class="stat-box"><h3>Image Space Saved</h3><div class="value" id="stat-images-saved">--</div></div>
  <div class="stat-box"><h3>Video Space Saved</h3><div class="value" id="stat-videos-saved">--</div></div>
</div>

<div class="tabs">
  <button class="tab active" data-tab="visual">🔍 Duplicates</button>
  <button class="tab" data-tab="folders">📁 By Folder</button>
  <button class="tab" data-tab="all-media">🖼️ Browse All</button>
  <button class="tab" data-tab="cleanup">🧹 Cleanup</button>
  <button class="tab" data-tab="history">📋 History</button>
  <button class="tab smart-tab" data-tab="smart-search">✨ AI Search</button>
</div>

<!-- Visual View -->
<div id="visual" class="tab-content active">
  <div class="action-bar">
    <div class="summary">
      <span id="selected-count">0</span> files marked for deletion
      (<span id="selected-size">0 B</span>)
    </div>
    <div class="controls">
      <select class="sort-select" id="sort-select">
        <option value="size_desc">Size (Largest First)</option>
        <option value="size_asc">Size (Smallest First)</option>
      </select>
      <button class="btn btn-lg btn-primary" id="btn-auto-select">Auto-Select Duplicates</button>
      <button class="btn btn-lg btn-danger" id="btn-delete-marked" disabled>Delete Marked</button>
      <button class="btn btn-lg btn-secondary" id="btn-reset-all">Reset All</button>
    </div>
  </div>
  <div id="groups-container"></div>
  <div class="pagination" id="pagination" style="display:none">
    <button id="btn-prev">&laquo; Previous</button>
    <span class="page-info">Page <span id="cur-page">1</span> of <span id="total-pages">1</span></span>
    <button id="btn-next">Next &raquo;</button>
  </div>
</div>

<!-- Folders -->
<div id="folders" class="tab-content">
  <h2 style="margin-bottom:6px">Folders Containing Duplicates</h2>
  <p style="color:#666;font-size:13px;margin-bottom:14px">Check the folders whose files you want to <strong>keep</strong>. Duplicates in all other folders will be deleted.</p>
  <div class="folder-actions">
    <div class="info"><span id="keep-folder-count">0</span> folder(s) marked as keep</div>
    <button class="btn btn-lg btn-primary" id="btn-preview-rules" disabled>Preview</button>
  </div>
  <div id="folder-preview" style="display:none"></div>
  <div class="folder-list" id="folder-list"></div>
</div>

<!-- All Media -->
<div id="all-media" class="tab-content">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap: wrap; gap: 10px;">
    <div>
      <h2 style="margin-bottom:2px">All Media Files</h2>
      <p style="color:#888;font-size:13px">Browse all indexed photos and videos. Click for full resolution.</p>
    </div>
    <div style="display:flex; align-items:center; gap:8px">
      <select class="sort-select" id="all-media-sort">
        <option value="size_desc">Largest First</option>
        <option value="size_asc">Smallest First</option>
      </select>
      <button class="btn btn-primary" id="btn-refresh-all-media">Refresh</button>
    </div>
  </div>

  <div class="action-bar" id="all-media-actions" style="display:none">
    <div class="summary">
      <span id="all-media-selected-count">0</span> files marked for deletion
      (<span id="all-media-selected-size">0 B</span>)
    </div>
    <div class="controls">
      <button class="btn btn-danger" id="btn-all-media-delete-marked" disabled>Delete Marked</button>
      <button class="btn btn-secondary" id="btn-all-media-reset">Reset</button>
    </div>
  </div>
  
  <div id="all-media-container" class="images" style="margin-top:20px;"></div>

  <div class="pagination" id="all-media-pagination" style="display:none; margin-top:20px">
    <button id="btn-all-media-prev">&laquo; Previous</button>
    <span class="page-info">Page <span id="all-media-cur-page">1</span> of <span id="all-media-total-pages">1</span></span>
    <button id="btn-all-media-next">Next &raquo;</button>
  </div>
</div>

<!-- Cleanup (Combined Empty Folders + Small Files) -->
<div id="cleanup" class="tab-content">
  <div class="cleanup-sub-tabs" style="display:flex; gap:8px; margin-bottom:20px; padding:4px; background:var(--bg-tertiary); border-radius:var(--radius-md); width:fit-content;">
    <button class="cleanup-sub-tab active" data-cleanup-tab="empty-folders-view">📂 Empty Folders</button>
    <button class="cleanup-sub-tab" data-cleanup-tab="small-files-view">📄 Small Files</button>
  </div>

  <!-- Empty Folders Sub-view -->
  <div id="empty-folders-view" class="cleanup-view active">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
      <div>
        <h2 style="margin-bottom:2px">Empty Folders</h2>
        <p style="color:var(--text-secondary);font-size:13px">Folders that contain no files (ignoring .DS_Store).</p>
      </div>
      <button class="btn btn-primary" id="btn-refresh-empty">Scan Now</button>
    </div>
    
    <div class="folder-actions" id="empty-folder-actions" style="display:none">
      <div class="info"><span id="empty-folder-selected-count">0</span> folder(s) selected</div>
      <button class="btn btn-lg btn-danger" id="btn-delete-empty" disabled>Delete Selected</button>
    </div>
    
    <div id="empty-folder-list" class="folder-list"></div>
  </div>

  <!-- Small Files Sub-view -->
  <div id="small-files-view" class="cleanup-view" style="display:none">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap: wrap; gap: 10px;">
      <div>
        <h2 style="margin-bottom:2px">Small Files / Screenshots</h2>
        <p style="color:var(--text-secondary);font-size:13px">Find and remove images smaller than a specific size.</p>
      </div>
      <div style="display:flex; align-items:center; gap:8px">
        <label style="font-size:13px; color:var(--text-secondary)">Min:</label>
        <input type="number" id="small-files-min-size" value="0.0" step="0.1" min="0" style="width:60px; padding:6px 10px; border:1px solid var(--border-color); border-radius:var(--radius-full); font-size:13px;">
        <label style="font-size:13px; color:var(--text-secondary)">Max (MB):</label>
        <input type="number" id="small-files-size" value="0.5" step="0.1" min="0" style="width:60px; padding:6px 10px; border:1px solid var(--border-color); border-radius:var(--radius-full); font-size:13px;">
        <button class="btn btn-primary" id="btn-refresh-small">Search</button>
      </div>
    </div>
    
    <div class="action-bar" id="small-files-actions" style="display:none">
      <div class="summary">
        <span id="small-selected-count">0</span> files marked for deletion
        (<span id="small-selected-size">0 B</span>)
      </div>
      <div class="controls">
        <label style="font-size:12px; cursor:pointer; margin-right:8px; color:var(--text-secondary)"><input type="checkbox" id="small-files-autoselect" checked> Auto-select</label>
        <button class="btn btn-secondary" id="btn-small-clear-kept">Clear Kept (<span id="small-kept-count">0</span>)</button>
        <button class="btn btn-primary" id="btn-small-select-all">Select All</button>
        <button class="btn btn-danger" id="btn-small-delete-marked" disabled>Delete Marked</button>
        <button class="btn btn-secondary" id="btn-small-reset">Reset</button>
      </div>
    </div>
    
    <div id="small-files-container" class="images" style="margin-top:20px;"></div>

    <div class="pagination" id="small-files-pagination" style="display:none; margin-top:20px">
      <button id="btn-small-prev">&laquo; Previous</button>
      <span class="page-info">Page <span id="small-cur-page">1</span> of <span id="small-total-pages">1</span></span>
      <button id="btn-small-next">Next &raquo;</button>
    </div>
  </div>
</div>

<!-- Smart Search -->
<div id="smart-search" class="tab-content">
  <div style="margin-bottom:20px; text-align:center;">
    <h2 style="margin-bottom:8px">&#10024; Smart AI Search</h2>
    <p style="color:#666;font-size:14px;margin-bottom:16px">Describe what you are looking for (e.g. "whiteboard", "receipt", "cat", "blur"). First run requires indexing.</p>
    
    <div style="max-width:600px; margin:0 auto; display:flex; gap:10px;">
        <input type="text" id="smart-query" placeholder="Enter search terms..." style="flex:1; padding:12px; border:1px solid #ccc; border-radius:8px; font-size:16px;">
        <button class="btn btn-primary" id="btn-smart-search" style="padding:10px 24px; font-size:16px;">Search</button>
    </div>
    <div style="margin-top:10px;">
        <button class="btn btn-secondary" id="btn-search-non-human" style="font-size:13px;" title="Finds images that don't contain people">Search Non-Human</button>
    </div>
    
    <div id="smart-status" style="margin-top:16px; font-size:13px; color:#555; background:#f9f9f9; padding:10px; border-radius:8px; display:inline-block;">
        Status: <span id="smart-status-text">Ready</span>
        <button class="btn btn-secondary" id="btn-smart-index" style="margin-left:10px; font-size:11px; padding:4px 8px;">Index All Images</button>
    </div>
    <div id="smart-progress-container" style="display:none; max-width:400px; margin:10px auto;">
        <div style="height:6px; background:#eee; border-radius:3px; overflow:hidden;">
            <div id="smart-progress-bar" style="height:100%; background:#4361ee; width:0%"></div>
        </div>
        <div style="font-size:11px; color:#888; margin-top:4px; text-align:right" id="smart-progress-text">0/0</div>
    </div>
  </div>

  <div class="action-bar" id="smart-actions" style="display:none">
    <div class="summary">
      <span id="smart-selected-count">0</span> files marked for deletion
    </div>
    <div class="controls">
        <button class="btn btn-primary" id="btn-smart-select-all">Select All</button>
        <button class="btn btn-danger" id="btn-smart-delete" disabled>Delete Selected</button>
    </div>
  </div>

  <div id="smart-results" class="images"></div>
</div>

<!-- History -->
<div id="history" class="tab-content">
  <div style="margin-bottom:24px">
    <h2 style="margin-bottom:8px">Deletion History</h2>
    <p style="color:#888;font-size:13px;margin-bottom:20px">Track all files you've deleted and the space you've recovered</p>
    
    <!-- Statistics Cards -->
    <div class="history-stats-grid">
      <div class="history-stat-card">
        <div class="stat-icon" style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%)">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
            <polyline points="17 21 17 13 7 13 7 21"/>
            <polyline points="7 3 7 8 15 8"/>
          </svg>
        </div>
        <div class="stat-content">
          <div class="stat-label">Total Space Saved</div>
          <div class="stat-value" id="hist-total-space">0 B</div>
        </div>
      </div>
      
      <div class="history-stat-card">
        <div class="stat-icon" style="background:linear-gradient(135deg, #f093fb 0%, #f5576c 100%)">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <polyline points="21 15 16 10 5 21"/>
          </svg>
        </div>
        <div class="stat-content">
          <div class="stat-label">Images Deleted</div>
          <div class="stat-value" id="hist-images-count">0</div>
          <div class="stat-sublabel" id="hist-images-space">0 B saved</div>
        </div>
      </div>
      
      <div class="history-stat-card">
        <div class="stat-icon" style="background:linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="23 7 16 12 23 17 23 7"/>
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
          </svg>
        </div>
        <div class="stat-content">
          <div class="stat-label">Videos Deleted</div>
          <div class="stat-value" id="hist-videos-count">0</div>
          <div class="stat-sublabel" id="hist-videos-space">0 B saved</div>
        </div>
      </div>
      
      <div class="history-stat-card">
        <div class="stat-icon" style="background:linear-gradient(135deg, #fa709a 0%, #fee140 100%)">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
        </div>
        <div class="stat-content">
          <div class="stat-label">Total Files</div>
          <div class="stat-value" id="hist-total-files">0</div>
          <div class="stat-sublabel">Across all folders</div>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Search and Filter Bar -->
  <div class="history-controls">
    <div class="history-search-box">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="11" cy="11" r="8"/>
        <path d="m21 21-4.35-4.35"/>
      </svg>
      <input type="text" id="history-search" placeholder="Search by filename or folder..." />
    </div>
    <div class="history-filter-group">
      <select id="history-filter" class="history-filter-select">
        <option value="all">All Files</option>
        <option value="images">Images Only</option>
        <option value="videos">Videos Only</option>
      </select>
      <select id="history-sort" class="history-filter-select">
        <option value="recent">Most Recent</option>
        <option value="oldest">Oldest First</option>
        <option value="size-desc">Largest First</option>
        <option value="size-asc">Smallest First</option>
      </select>
    </div>
    <button class="btn btn-secondary" onclick="loadHistory()" style="white-space:nowrap">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right:6px">
        <polyline points="23 4 23 10 17 10"/>
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
      </svg>
      Refresh
    </button>
  </div>
  
  <div id="history-list"></div>
</div>

<script>
(function(){
'use strict';

// State
let currentPage = 1;
let currentSort = 'size_desc';
let totalPages = 1;
const markedForDeletion = new Map(); // path -> size

// ----- Helpers -----
function formatSize(bytes){
  const u=['B','KB','MB','GB','TB'];
  let i=0;
  while(bytes>=1024&&i<u.length-1){bytes/=1024;i++}
  return bytes.toFixed(1)+' '+u[i];
}

function escapeHtml(s){
  const d=document.createElement('div');d.textContent=s;return d.innerHTML;
}

function showToast(msg,type='success'){
  const c=document.getElementById('toast-container');
  const t=document.createElement('div');
  t.className='toast '+type;
  t.textContent=msg;
  c.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';t.style.transition='opacity .3s';setTimeout(()=>t.remove(),300)},3500);
}

// ----- API calls -----
async function apiGet(url){
  const r=await fetch(url);
  return r.json();
}
async function apiPost(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return r.json();
}

// ----- Scan polling -----
async function pollScan(){
  const overlay=document.getElementById('scan-overlay');
  const phaseEl=document.getElementById('scan-phase');
  overlay.style.display='flex';
  while(true){
    try{
      const s=await apiGet('/api/stats');
      if(s.scan_complete){
        overlay.style.display='none';
        refreshAll();
        return;
      }
      phaseEl.textContent=s.scan_detail||s.scan_phase||'Scanning...';
    }catch(e){}
    await new Promise(r=>setTimeout(r,800));
  }
}

// ----- Refresh everything -----
async function refreshAll(){
  console.log("Refreshing UI data...");
  const container = document.getElementById('groups-container');
  // Loading state for the main list
  container.innerHTML = '<div class="empty"><div class="spinner"></div><p>Loading your duplicate groups...</p></div>';
  
  try {
      await Promise.all([refreshStats(), loadGroups(), loadFolders(), loadHistory()]);
      console.log("Refresh complete.");
  } catch (e) {
      console.error("Manual refresh failed:", e);
      showToast("Refresh failed: " + e.message, "error");
  }
}

async function refreshStats(){
  try {
      const s = await apiGet('/api/stats');
      document.getElementById('stat-groups').textContent = s.total_groups ?? 0;
      document.getElementById('stat-wasted').textContent = s.wasted_space_human ?? '0 B';
      document.getElementById('stat-saved').textContent = s.space_saved_human ?? '0 B';
      document.getElementById('stat-images-deleted').textContent = s.images_deleted ?? 0;
      document.getElementById('stat-videos-deleted').textContent = s.videos_deleted ?? 0;
      document.getElementById('stat-images-saved').textContent = s.space_saved_images_human ?? '0 B';
      document.getElementById('stat-videos-saved').textContent = s.space_saved_videos_human ?? '0 B';
  } catch (e) {
      console.error("Stats fetch failed:", e);
      throw e;
  }
}

// ----- Groups -----
async function loadGroups(){
  const data=await apiGet(`/api/groups?page=${currentPage}&sort=${currentSort}&per_page=50`);
  totalPages=data.total_pages;
  document.getElementById('cur-page').textContent=data.page;
  document.getElementById('total-pages').textContent=data.total_pages;
  const pag=document.getElementById('pagination');
  pag.style.display=data.total_groups>0?'flex':'none';
  document.getElementById('btn-prev').disabled=data.page<=1;
  document.getElementById('btn-next').disabled=data.page>=data.total_pages;

  const container=document.getElementById('groups-container');
  if(data.groups.length===0){
    container.innerHTML='<div class="empty"><div class="icon">&#10004;</div><p>No duplicate groups to show.</p></div>';
    return;
  }

  let html='';
  data.groups.forEach((g,gi)=>{
    const groupNum=(data.page-1)*data.per_page+gi+1;
    html+=`<div class="group" data-hash="${g.hash}">`;
    html+=`<div class="group-header" data-group-toggle="${g.hash}"><h3><span class="group-toggle-icon">&#9660;</span> Group #${groupNum} &mdash; ${g.count} files, ${g.file_size_human} each</h3>`;
    html+=`<button class="btn btn-secondary" data-reset-group="${g.hash}">Reset</button></div>`;
    html+=`<div class="group-body" data-group-body="${g.hash}" style="display: block"><div class="images">`;
    g.files.forEach(f=>{
      const isMarked=markedForDeletion.has(f.path);
      const cls=isMarked?'imgbox delete-mark':'imgbox';
      html+=`<div class="${cls}" data-path="${escapeHtml(f.path)}" data-size="${f.size}">`;
      html+=`<div class="thumb-container">`;
      html+=`<img src="/thumbnails/${f.thumbnail}" loading="lazy" alt="">`;
      if(f.type==='video') html+=`<div class="video-overlay">&#9654;</div>`;
      html+=`</div>`;
      html+=`<div class="path" title="${escapeHtml(f.path)}">${escapeHtml(f.path)}</div>`;
      const badge=f.type==='video'?'<span class="type-badge video">VIDEO</span>':'<span class="type-badge photo">PHOTO</span>';
      html+=`<div class="meta">${f.size_human} ${badge}</div>`;
      html+=`<div class="actions">`;
      html+=`<button class="btn btn-keep btn-action" data-action="keep">Keep</button>`;
      html+=`<button class="btn btn-delete btn-action" data-action="delete-mark">Mark Delete</button>`;
      html+=`<button class="btn btn-danger btn-action" data-action="delete-now">Delete Now</button>`;
      html+=`</div></div>`;
    });
    html+=`</div></div></div>`;
  });
  container.innerHTML=html;
}

// ----- Folders -----
const keepFolders=new Set();
let allFoldersData=[];  // full list from /api/folders (used when no keep selection)
let folderListHasDeletableFiles=false;  // true when filtered list has files to delete (after keep selection)

async function loadFolders(){
  const data=await apiGet('/api/folders');
  allFoldersData=data||[];
  await refreshFolderList();
  updateKeepCount();
}

async function refreshFolderList(){
  const el=document.getElementById('folder-list');
  if(!allFoldersData.length){el.innerHTML='<div class="empty"><p>No folders with duplicates.</p></div>';return}

  let deleteMap = new Map(); // folder -> {count, size_human}
  folderListHasDeletableFiles = false;

  if (keepFolders.size > 0) {
      try {
          const resp = await apiPost('/api/folder-rules', {keep_folders: [...keepFolders], dry_run: true});
          if (resp.by_folder) {
              for (const [folder, files] of Object.entries(resp.by_folder)) {
                  const size = files.reduce((s, f) => s + f.size, 0);
                  deleteMap.set(folder, {count: files.length, size_human: formatSize(size)});
                  folderListHasDeletableFiles = true;
              }
          }
      } catch (e) { console.error("Rules check failed", e); }
  }

  el.innerHTML = allFoldersData.map(f => {
    const isKept = keepFolders.has(f.folder);
    const delInfo = deleteMap.get(f.folder);
    
    let cls = 'folder-item';
    let badgeHtml = `<span class="folder-count">${f.count} dups &middot; ${f.size_human}</span>`;
    
    if (isKept) {
        cls += ' kept';
        badgeHtml = `<span class="folder-count keep-badge">KEEP &middot; ${f.count} files</span>`;
    } else if (delInfo) {
        cls += ' to-delete';
        badgeHtml = `<span class="folder-count delete-badge">DELETE ${delInfo.count} files &middot; ${delInfo.size_human}</span>`;
    }

    return `<div class="${cls}">
        <input type="checkbox" data-keep-folder="${escapeHtml(f.folder)}" ${isKept ? 'checked' : ''}>
        <span class="folder-path">${escapeHtml(f.folder)}</span>
        ${badgeHtml}
    </div>`;
  }).join('');
}

function updateKeepCount(){
  document.getElementById('keep-folder-count').textContent=keepFolders.size;
  const hasSelection=keepFolders.size>0;
  document.getElementById('btn-preview-rules').disabled=!hasSelection||!folderListHasDeletableFiles;
  if(!hasSelection){
    document.getElementById('folder-preview').style.display='none';
  }
}

// ----- Visual Preview state -----
let previewChecked = new Map(); // path -> {size, checked}
let previewData = null; // last response from dry_run

function updatePreviewCounts(){
  let count=0, totalSize=0, totalFiles=0;
  previewChecked.forEach(v=>{
    totalFiles++;
    if(v.checked){count++;totalSize+=v.size}
  });
  const infoEl=document.querySelector('.pbb-info');
  if(infoEl) infoEl.innerHTML=`<strong>${count}</strong> of ${totalFiles} files selected for deletion (<strong>${formatSize(totalSize)}</strong>)`;
  const confirmBtn=document.getElementById('btn-confirm-preview-delete');
  if(confirmBtn){
    confirmBtn.textContent=`Confirm Delete ${count} Files`;
    confirmBtn.disabled=count===0;
  }
}

function renderVisualPreview(resp){
  previewData=resp;
  previewChecked.clear();
  const byFolder=resp.by_folder;
  const folderNames=Object.keys(byFolder).sort();
  const totalFolders=folderNames.length;

  // Register all files as checked
  folderNames.forEach(folder=>{
    byFolder[folder].forEach(f=>{
      previewChecked.set(f.path,{size:f.size,checked:true});
    });
  });

  // Hide folder list + actions, show preview area
  document.getElementById('folder-list').style.display='none';
  document.querySelector('.folder-actions').style.display='none';

  const el=document.getElementById('folder-preview');
  let html='';

  // Summary bar
  html+=`<div class="preview-summary-bar"><strong>${resp.files_to_delete} files</strong> (${resp.total_size_human}) will be deleted from <strong>${totalFolders} folders</strong> across ${resp.groups_affected} duplicate groups</div>`;

  // Per-folder sections
  folderNames.forEach(folder=>{
    const files=byFolder[folder];
    const folderSize=files.reduce((s,f)=>s+f.size,0);
    html+=`<div class="preview-folder-section" data-pf="${escapeHtml(folder)}">`;
    html+=`<div class="preview-folder-header" data-pf-toggle="${escapeHtml(folder)}">`;
    html+=`<span class="folder-toggle open">&#9654;</span>`;
    html+=`<span class="folder-name">${escapeHtml(folder)}</span>`;
    html+=`<span class="folder-badge">${files.length} files &middot; ${formatSize(folderSize)}</span>`;
    html+=`<button class="folder-check-toggle" data-pf-check="${escapeHtml(folder)}">Uncheck All</button>`;
    html+=`</div>`;
    html+=`<div class="preview-folder-body" data-pf-body="${escapeHtml(folder)}">`;
    html+=`<div class="preview-grid">`;
    files.forEach(f=>{
      const fname=f.path.split('/').pop();
      html+=`<div class="preview-card" data-pp="${escapeHtml(f.path)}" data-ps="${f.size}">`;
      html+=`<input type="checkbox" class="pc-checkbox" checked data-pc="${escapeHtml(f.path)}">`;
      html+=`<img class="pc-thumb" src="/thumbnails/${f.thumbnail}" loading="lazy" alt="">`;
      if(f.type==='video') html+=`<div class="video-badge">&#9654;</div>`;
      html+=`<div class="pc-name" title="${escapeHtml(f.path)}">${escapeHtml(fname)}</div>`;
      html+=`<div class="pc-meta">${f.size_human}</div>`;
      html+=`</div>`;
    });
    html+=`</div></div></div>`;
  });

  // Sticky bottom bar
  html+=`<div class="preview-bottom-bar">`;
  html+=`<div class="pbb-info"></div>`;
  html+=`<div class="pbb-actions">`;
  html+=`<button class="btn btn-lg btn-secondary" id="btn-back-preview">Back</button>`;
  html+=`<button class="btn btn-lg btn-danger" id="btn-confirm-preview-delete">Confirm Delete</button>`;
  html+=`</div></div>`;

  // Extra bottom padding so content isn't hidden behind sticky bar
  html+=`<div style="height:70px"></div>`;

  el.innerHTML=html;
  el.style.display='block';
  updatePreviewCounts();
}

function closeVisualPreview(){
  document.getElementById('folder-preview').style.display='none';
  document.getElementById('folder-preview').innerHTML='';
  document.getElementById('folder-list').style.display='';
  document.querySelector('.folder-actions').style.display='';
  previewChecked.clear();
  previewData=null;
}

async function previewFolderRules(){
  const btn=document.getElementById('btn-preview-rules');
  btn.textContent='Loading...';btn.disabled=true;
  const resp=await apiPost('/api/folder-rules',{keep_folders:[...keepFolders],dry_run:true});
  btn.textContent='Preview';btn.disabled=false;

  if(resp.files_to_delete===0){
    const el=document.getElementById('folder-preview');
    el.innerHTML='<div class="preview-box">No files to delete. Either no duplicates span these folders, or all copies are in keep folders.</div>';
    el.style.display='block';
    document.getElementById('btn-execute-rules').style.display='none';
    return;
  }

  renderVisualPreview(resp);
}

async function executePreviewDelete(){
  const paths=[];
  previewChecked.forEach((v,path)=>{if(v.checked)paths.push(path)});
  if(!paths.length)return;
  if(!confirm(`Move ${paths.length} file(s) to Trash?`))return;

  const btn=document.getElementById('btn-confirm-preview-delete');
  btn.textContent=`Deleting ${paths.length}...`;btn.disabled=true;

  const resp=await apiPost('/api/delete-batch',{paths});
  if(resp.succeeded>0){
    showToast(`Deleted ${resp.succeeded} file(s), freed ${resp.total_freed_human}`);
  }
  if(resp.failed>0){
    showToast(`${resp.failed} file(s) failed to delete`,'error');
  }

  closeVisualPreview();
  await refreshAll();
}

// ----- History -----
function getFolderFromPath(path){
  const i=Math.max(path.lastIndexOf('/'),path.lastIndexOf('\\'));
  return i<=0?path:path.slice(0,i);
}
async function loadHistory() {
    const el = document.getElementById('history-list');
    el.innerHTML = '<div class="empty"><div class="spinner"></div><p>Calculating file locations...</p></div>';

    try {
        const data = await apiGet('/api/history?enrich=true');

        // Update statistics cards
        const totalSpace = data.total_space_saved || 0;
        const totalFiles = (data.deletions || []).length;

        // Count images and videos
        let imageCount = 0;
        let videoCount = 0;
        let imageSpace = 0;
        let videoSpace = 0;

        (data.deletions || []).forEach(d => {
            const ext = (d.path || '').toLowerCase();
            const isVideo = ext.endsWith('.mp4') || ext.endsWith('.mov') || ext.endsWith('.avi') ||
                ext.endsWith('.mkv') || ext.endsWith('.m4v') || ext.endsWith('.wmv') ||
                ext.endsWith('.flv') || ext.endsWith('.webm') || ext.endsWith('.3gp');

            if (isVideo) {
                videoCount++;
                videoSpace += (d.size || 0);
            } else {
                imageCount++;
                imageSpace += (d.size || 0);
            }
        });

        document.getElementById('hist-total-space').textContent = formatSize(totalSpace);
        document.getElementById('hist-images-count').textContent = imageCount.toLocaleString();
        document.getElementById('hist-images-space').textContent = formatSize(imageSpace) + ' saved';
        document.getElementById('hist-videos-count').textContent = videoCount.toLocaleString();
        document.getElementById('hist-videos-space').textContent = formatSize(videoSpace) + ' saved';
        document.getElementById('hist-total-files').textContent = totalFiles.toLocaleString();

        if (!data.deletions || !data.deletions.length) {
            el.innerHTML = '<div class="empty"><p>No files deleted yet.</p></div>';
            return;
        }

        const items = [...data.deletions].reverse();
        const byFolder = new Map();
        items.forEach(d => {
            const folder = getFolderFromPath(d.path);
            if (!byFolder.has(folder)) byFolder.set(folder, []);
            byFolder.get(folder).push(d);
        });

        const folderOrder = [...byFolder.keys()];
        el.innerHTML = folderOrder.map((folder, i) => {
            const files = byFolder.get(folder);
            const fileCount = files.length;
            const folderSize = files.reduce((s, d) => s + (d.size || 0), 0);
            const sizeHuman = formatSize(folderSize);
            const sectionId = 'hist-section-' + i;

            let body = files.map(d => {
                const dt = new Date(d.timestamp);
                const dateStr = dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
                const timeStr = dt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
                const fname = d.path ? d.path.split(/[/\\\\]/).pop() : 'Unknown';

                // Determine file type
                const ext = (d.path || '').toLowerCase();
                const isVideo = ext.endsWith('.mp4') || ext.endsWith('.mov') || ext.endsWith('.avi') ||
                    ext.endsWith('.mkv') || ext.endsWith('.m4v') || ext.endsWith('.wmv') ||
                    ext.endsWith('.flv') || ext.endsWith('.webm') || ext.endsWith('.3gp');

                const iconClass = isVideo ? 'video' : 'photo';
                const iconSvg = isVideo ?
                    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>' :
                    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>';

                let keptHtml = '';
                if (d.kept_versions && d.kept_versions.length > 0) {
                    const keptPath = d.kept_versions[0];
                    const keptName = keptPath.split(/[/\\\\]/).pop();
                    const keptFolder = getFolderFromPath(keptPath);
                    const moreCount = d.kept_versions.length - 1;
                    const moreText = moreCount > 0 ? ` (+${moreCount} more)` : '';

                    keptHtml = `<div class="kept-info">
                      <strong>✓ Kept Original:</strong> ${escapeHtml(keptName)}${moreText}<br>
                      <small style="opacity:0.8">📁 ${escapeHtml(keptFolder)}</small>
                  </div>`;
                }

                return `<div class="history-item" data-file-type="${iconClass}">
                <div class="history-item-icon ${iconClass}">${iconSvg}</div>
                <div class="history-item-content">
                    <span class="hist-path" title="${escapeHtml(d.path)}">${escapeHtml(fname)}</span>
                    ${keptHtml}
                </div>
                <div class="hist-meta">
                    <span class="hist-size">${d.size_human || ''}</span>
                    <span class="hist-timestamp">${dateStr}<br>${timeStr}</span>
                </div>
              </div>`;
            }).join('');

            return `<div class="history-folder-section" data-hist-section="${sectionId}">
              <div class="history-folder-header" data-hist-toggle="${sectionId}">
                  <span class="hist-folder-toggle">&#9654;</span>
                  <span class="hist-folder-label">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right:6px">
                          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                      </svg>
                      ${escapeHtml(folder)} &middot; ${fileCount} file(s) &middot; ${sizeHuman}
                  </span>
              </div>
              <div class="history-folder-body collapsed" data-hist-body="${sectionId}">
                  ${body}
              </div>
          </div>`;
        }).join('');

        // Add search and filter event listeners
        setupHistoryFilters();
    } catch (e) {
        console.error("History load failed:", e);
        el.innerHTML = `<div class="empty" style="color:#e74c3c"><p>Failed to load deletion history.</p></div>`;
    }
}

function setupHistoryFilters() {
    const searchInput = document.getElementById('history-search');
    const filterSelect = document.getElementById('history-filter');
    const sortSelect = document.getElementById('history-sort');

    if (!searchInput || !filterSelect || !sortSelect) return;

    const applyFilters = () => {
        const searchTerm = searchInput.value.toLowerCase();
        const filterType = filterSelect.value;
        const sortType = sortSelect.value;

        const sections = document.querySelectorAll('.history-folder-section');
        sections.forEach(section => {
            const items = section.querySelectorAll('.history-item');
            let visibleCount = 0;

            items.forEach(item => {
                const path = item.querySelector('.hist-path').textContent.toLowerCase();
                const fileType = item.getAttribute('data-file-type');

                let show = true;

                // Apply search filter
                if (searchTerm && !path.includes(searchTerm)) {
                    show = false;
                }

                // Apply type filter
                if (filterType === 'images' && fileType !== 'photo') show = false;
                if (filterType === 'videos' && fileType !== 'video') show = false;

                item.style.display = show ? 'flex' : 'none';
                if (show) visibleCount++;
            });

            // Hide section if no visible items
            section.style.display = visibleCount > 0 ? 'block' : 'none';
        });
    };

    searchInput.addEventListener('input', applyFilters);
    filterSelect.addEventListener('change', applyFilters);
    sortSelect.addEventListener('change', applyFilters);
}


// ----- Empty Folders -----
const selectedEmptyFolders = new Set();

async function loadEmptyFolders() {
    const el = document.getElementById('empty-folder-list');
    el.innerHTML = '<div class="empty"><div class="spinner"></div><p>Searching for empty folders...</p></div>';
    document.getElementById('empty-folder-actions').style.display = 'none';
    selectedEmptyFolders.clear();
    updateEmptyFolderCount();
    
    try {
        const folders = await apiGet('/api/empty-folders');
        if (!folders || folders.length === 0) {
            el.innerHTML = '<div class="empty"><div class="icon">&#10004;</div><p>No empty folders found.</p></div>';
            return;
        }
        
        document.getElementById('empty-folder-actions').style.display = 'flex';
        
        el.innerHTML = folders.map(path => {
            return `<div class="folder-item">
                <input type="checkbox" data-empty-folder="${escapeHtml(path)}">
                <span class="folder-path">${escapeHtml(path)}</span>
            </div>`;
        }).join('');
    } catch (e) {
        console.error("Empty folders load failed:", e);
        el.innerHTML = `<div class="empty" style="color:#e74c3c"><p>Failed to load empty folders.</p></div>`;
    }
}

function updateEmptyFolderCount() {
    document.getElementById('empty-folder-selected-count').textContent = selectedEmptyFolders.size;
    document.getElementById('btn-delete-empty').disabled = selectedEmptyFolders.size === 0;
}

async function deleteSelectedEmptyFolders() {
    const paths = [...selectedEmptyFolders];
    if (paths.length === 0) return;
    if (!confirm(`Permanently delete ${paths.length} empty folder(s)?`)) return;
    
    const btn = document.getElementById('btn-delete-empty');
    const originalText = btn.textContent;
    btn.textContent = 'Deleting...';
    btn.disabled = true;
    
    try {
        const resp = await apiPost('/api/delete-empty-folders', { paths });
        const succeeded = resp.results.filter(r => r.success).length;
        const failed = resp.results.filter(r => !r.success).length;
        
        if (succeeded > 0) showToast(`Deleted ${succeeded} folder(s)`);
        if (failed > 0) showToast(`Failed to delete ${failed} folder(s)`, 'error');
        
        await loadEmptyFolders();
    } catch (e) {
        console.error("Delete empty folders failed:", e);
        showToast("Error deleting folders", "error");
    } finally {
        btn.textContent = originalText;
    }
}

// ----- Mark / unmark -----
function updateSummary(){
  let total=0;
  markedForDeletion.forEach(s=>{total+=s});
  document.getElementById('selected-count').textContent=markedForDeletion.size;
  document.getElementById('selected-size').textContent=formatSize(total);
  document.getElementById('btn-delete-marked').disabled=markedForDeletion.size===0;
}

// ----- Single delete -----
async function deleteFile(path){
  const box=document.querySelector(`.imgbox[data-path="${CSS.escape(path)}"]`);
  if(box){
    const btn=box.querySelector('[data-action="delete-now"]');
    if(btn){btn.textContent='Deleting...';btn.disabled=true}
  }
  const resp=await apiPost('/api/delete',{path});
  if(resp.success){
    markedForDeletion.delete(path);
    updateSummary();
    showToast(`Moved to Trash (${resp.size_freed_human} freed)`);
    await Promise.all([refreshStats(),loadGroups(),loadHistory()]);
  }else{
    showToast('Delete failed: '+(resp.error||'Unknown error'),'error');
    if(box){
      const btn=box.querySelector('[data-action="delete-now"]');
      if(btn){btn.textContent='Delete Now';btn.disabled=false}
    }
  }
}

// ----- Batch delete -----
async function batchDelete(){
  const paths=[...markedForDeletion.keys()];
  if(!paths.length)return;
  if(!confirm(`Move ${paths.length} file(s) to Trash?`))return;
  const btn=document.getElementById('btn-delete-marked');
  btn.textContent=`Deleting ${paths.length}...`;btn.disabled=true;
  const resp=await apiPost('/api/delete-batch',{paths});
  markedForDeletion.clear();
  updateSummary();
  btn.textContent='Delete Marked';
  if(resp.succeeded>0){
    showToast(`Deleted ${resp.succeeded} file(s), freed ${resp.total_freed_human}`);
  }
  if(resp.failed>0){
    showToast(`${resp.failed} file(s) failed to delete`,'error');
  }
  await Promise.all([refreshStats(),loadGroups(),loadFolders(),loadHistory()]);
}

// ----- Auto-select -----
function autoSelectPage(){
  document.querySelectorAll('.group').forEach(group=>{
    const boxes=group.querySelectorAll('.imgbox');
    boxes.forEach((box,i)=>{
      const path=box.dataset.path;
      const size=parseInt(box.dataset.size)||0;
      if(i===0){
        box.classList.add('keep');box.classList.remove('delete-mark');
        markedForDeletion.delete(path);
      }else{
        box.classList.remove('keep');box.classList.add('delete-mark');
        markedForDeletion.set(path,size);
      }
    });
  });
  updateSummary();
}

function resetAll(){
  document.querySelectorAll('.imgbox').forEach(b=>{
    b.classList.remove('keep','delete-mark');
  });
  markedForDeletion.clear();
  updateSummary();
}

// ---------------------------------------------------------------------------
// Selection Mode & Bulk Actions
// ---------------------------------------------------------------------------
let isSelectionMode = false;
let lastSelectedObj = null; // { containerId, index, path }

function toggleSelectionMode() {
    isSelectionMode = !isSelectionMode;
    document.body.classList.toggle('selection-mode', isSelectionMode);
    const btn = document.getElementById('btn-toggle-selection-mode');
    if (isSelectionMode) {
        btn.classList.add('btn-primary');
        btn.classList.remove('btn-secondary');
        btn.innerHTML = '<span id="selection-mode-icon">✅</span> Exit Edit';
        showToast("Edit Mode: Click to select, Shift+Click for range");
    } else {
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
        btn.innerHTML = '<span id="selection-mode-icon">🔳</span> Bulk Edit';
    }
}

function handleItemClick(e, box, containerId) {
    const path = box.dataset.path || box.dataset.allPath || box.dataset.smallPath || box.dataset.smartPath;
    const size = parseInt(box.dataset.size) || 0;
    const container = document.getElementById(containerId);
    const allBoxes = Array.from(container.querySelectorAll('.imgbox'));
    const currentIndex = allBoxes.indexOf(box);

    // Helper to toggle mark
    const toggleMark = (targetBox, shouldMark) => {
        const itemPath = targetBox.dataset.path || targetBox.dataset.allPath || targetBox.dataset.smallPath || targetBox.dataset.smartPath;
        const itemSize = parseInt(targetBox.dataset.size) || 0;
        
        // Use the appropriate map based on containerId
        let targetMap = markedForDeletion;
        if(containerId === 'all-media-container') targetMap = markedAllMedia;
        else if(containerId === 'small-files-container') targetMap = markedSmallFiles;
        else if(containerId === 'smart-results') targetMap = smartSelected;

        if (shouldMark === undefined) shouldMark = !targetMap.has(itemPath);

        if (shouldMark) {
            targetBox.classList.add('delete-mark');
            if (targetMap instanceof Set) targetMap.add(itemPath);
            else targetMap.set(itemPath, itemSize);
            if (containerId === 'small-files-container') keptSmallFiles.delete(itemPath);
        } else {
            targetBox.classList.remove('delete-mark');
            targetMap.delete(itemPath);
            if (containerId === 'small-files-container') keptSmallFiles.add(itemPath);
        }
    };

    if (e.shiftKey && lastSelectedObj && lastSelectedObj.containerId === containerId) {
        const start = Math.min(lastSelectedObj.index, currentIndex);
        const end = Math.max(lastSelectedObj.index, currentIndex);
        const alreadyMarked = markedForDeletion.has(path) || markedAllMedia.has(path) || markedSmallFiles.has(path) || smartSelected.has(path);
        
        for (let i = start; i <= end; i++) {
            toggleMark(allBoxes[i], true);
        }
    } else {
        toggleMark(box);
    }

    lastSelectedObj = { containerId, index: currentIndex, path };
    
    // Update relevant summaries
    if(containerId === 'all-media-container') updateAllMediaSummary();
    else if(containerId === 'small-files-container') updateSmallSelectedCount();
    else if(containerId === 'smart-results') updateSmartActions();
    else updateSummary();
}

// ----- Event delegation -----
document.addEventListener('click',function(e){
  // Cache closest imgbox for multi-use
  const box = e.target.closest('.imgbox');

  // 1. Group Collapse/Expand (Priority)
  // Check closest header, ignore button clicks inside it
  const groupHeader = e.target.closest('[data-group-toggle]');
  if(groupHeader && !e.target.closest('button')){
      const hash = groupHeader.dataset.groupToggle;
      const group = groupHeader.closest('.group');
      const body = group.querySelector('.group-body');
      const arrow = groupHeader.querySelector('.group-toggle-icon');
      
      if(body){
          if(body.style.display !== 'none'){
              body.style.display = 'none';
              if(arrow) arrow.style.transform = 'rotate(-90deg)';
          } else {
              body.style.display = 'block';
              if(arrow) arrow.style.transform = 'rotate(0deg)';
          }
      }
      return;
  }

  // 2. Tab switching
  if(e.target.matches('.tab') || e.target.closest('.tab')){
    const tabBtn = e.target.closest('.tab');
    if(!tabBtn || !tabBtn.dataset.tab) return;
    
    const id = tabBtn.dataset.tab;
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
    
    tabBtn.classList.add('active');
    const content = document.getElementById(id);
    if(content) content.classList.add('active');
    
    // Refresh data if switching to specific tabs
    if(id === 'history') loadHistory();
    if(id === 'cleanup') {
      // Check which sub-tab is active and load accordingly
      const activeSubTab = document.querySelector('.cleanup-sub-tab.active');
      if(activeSubTab && activeSubTab.dataset.cleanupTab === 'small-files-view') {
        loadSmallFiles();
      } else {
        loadEmptyFolders();
      }
    }
    if(id === 'all-media') loadAllMedia();
    if(id === 'smart-search') checkSmartStatus();
    return;
  }

  // Cleanup sub-tab switching
  if(e.target.matches('.cleanup-sub-tab')){
    const subTabId = e.target.dataset.cleanupTab;
    document.querySelectorAll('.cleanup-sub-tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.cleanup-view').forEach(v=>v.classList.remove('active'));
    
    e.target.classList.add('active');
    const view = document.getElementById(subTabId);
    if(view) {
      view.classList.add('active');
      view.style.display = 'block';
    }
    
    // Hide the other view
    document.querySelectorAll('.cleanup-view:not(.active)').forEach(v=>v.style.display = 'none');
    
    // Load data for the selected sub-tab
    if(subTabId === 'empty-folders-view') loadEmptyFolders();
    if(subTabId === 'small-files-view') loadSmallFiles();
    return;
  }

  // Modal close
  if(e.target.matches('.modal-close') || e.target.matches('.modal')){
    closeModal();
    return;
  }

  // Handle imgbox clicks (Main Logic for Select/Preview)
  if(box && !e.target.closest('button')) {
    // Determine which container we are in
    const container = box.closest('.images');
    const containerId = container ? container.id : null;

    if (isSelectionMode) {
      if(containerId) handleItemClick(e, box, containerId);
    } else {
      // If NOT in selection mode, clicking opens full res
      const path = box.dataset.fullPath || box.dataset.path || box.dataset.allPath || box.dataset.smallPath || box.dataset.smartPath;
      const type = box.dataset.type || (box.querySelector('.type-badge.video') ? 'video' : 'photo');
      if(path) openFullRes(path, type);
    }
    return;
  }

  // 3. Group Actions (Keep, Delete, Reset)
  if(e.target.matches('[data-action="keep"]') && box){
    box.classList.add('keep');box.classList.remove('delete-mark');
    markedForDeletion.delete(box.dataset.path);
    updateSummary();
    return;
  }
  if(e.target.matches('[data-action="delete-mark"]') && box){
    box.classList.remove('keep');box.classList.add('delete-mark');
    markedForDeletion.set(box.dataset.path,parseInt(box.dataset.size)||0);
    updateSummary();
    return;
  }
  if(e.target.matches('[data-action="delete-now"]') && box){
    deleteFile(box.dataset.path);
    return;
  }
  if(e.target.matches('[data-reset-group]')){
    const hash = e.target.dataset.resetGroup;
    document.querySelectorAll(`.group[data-hash="${hash}"] .imgbox`).forEach(b=>{
      b.classList.remove('keep','delete-mark');
      markedForDeletion.delete(b.dataset.path);
    });
    updateSummary();
    return;
  }

  // 4. Visual preview: folder header toggle
  const pfHeader = e.target.closest('[data-pf-toggle]');
  if(pfHeader && !e.target.matches('.folder-check-toggle')){
    const folder = pfHeader.dataset.pfToggle;
    const body = document.querySelector(`[data-pf-body="${CSS.escape(folder)}"]`);
    const arrow = pfHeader.querySelector('.folder-toggle');
    if(body){
        body.classList.toggle('collapsed');
        if(arrow) arrow.classList.toggle('open');
    }
    return;
  }

  // 5. Visual preview: check/uncheck all in folder
  if(e.target.matches('[data-pf-check]')){
    e.stopPropagation();
    const folder=e.target.dataset.pfCheck;
    const body=document.querySelector(`[data-pf-body="${CSS.escape(folder)}"]`);
    if(!body)return;
    const boxes=body.querySelectorAll('.pc-checkbox');
    const allChecked=[...boxes].every(cb=>cb.checked);
    boxes.forEach(cb=>{
      cb.checked=!allChecked;
      const path=cb.dataset.pc;
      const card=cb.closest('.preview-card');
      const entry=previewChecked.get(path);
      if(entry)entry.checked=!allChecked;
      if(card){if(!allChecked)card.classList.remove('unchecked');else card.classList.add('unchecked')}
    });
    e.target.textContent=allChecked?'Check All':'Uncheck All';
    updatePreviewCounts();
    return;
  }

  // 6. Visual preview: back button
  if(e.target.id==='btn-back-preview'){
    closeVisualPreview();
    return;
  }
  // 7. Visual preview: confirm delete
  if(e.target.id==='btn-confirm-preview-delete'){
    executePreviewDelete();
    return;
  }
  // 8. Toggle Selection Mode
  if(e.target.id === 'btn-toggle-selection-mode' || e.target.closest('#btn-toggle-selection-mode')){
    toggleSelectionMode();
    return;
  }
});

// Buttons
document.getElementById('btn-auto-select').addEventListener('click',autoSelectPage);
document.getElementById('btn-delete-marked').addEventListener('click',batchDelete);
document.getElementById('btn-reset-all').addEventListener('click',resetAll);
document.getElementById('sort-select').addEventListener('change',function(){
  currentSort=this.value;currentPage=1;loadGroups();
});
document.getElementById('btn-prev').addEventListener('click',function(){
  if(currentPage>1){currentPage--;loadGroups();window.scrollTo(0,0)}
});
document.getElementById('btn-next').addEventListener('click',function(){
  if(currentPage<totalPages){currentPage++;loadGroups();window.scrollTo(0,0)}
});
// History folder collapse (delegated)
document.getElementById('history-list').addEventListener('click',function(e){
  const header=e.target.closest('[data-hist-toggle]');
  if(header){
    const section=header.closest('.history-folder-section');
    const body=section?section.querySelector('.history-folder-body'):null;
    const arrow=header.querySelector('.hist-folder-toggle');
    if(body){body.classList.toggle('collapsed');if(arrow)arrow.classList.toggle('open')}
  }
});
// Folder keep checkboxes (delegated)
document.getElementById('folder-list').addEventListener('change',async function(e){
  if(e.target.matches('[data-keep-folder]')){
    const folder=e.target.dataset.keepFolder;
    const item=e.target.closest('.folder-item');
    if(e.target.checked){keepFolders.add(folder);item.classList.add('kept')}
    else{keepFolders.delete(folder);item.classList.remove('kept')}
    closeVisualPreview();
    await refreshFolderList();
    updateKeepCount();
  }
});
// Preview card checkboxes (delegated on folder-preview container)
document.getElementById('folder-preview').addEventListener('change',function(e){
  if(e.target.matches('.pc-checkbox')){
    const path=e.target.dataset.pc;
    const card=e.target.closest('.preview-card');
    const entry=previewChecked.get(path);
    if(entry)entry.checked=e.target.checked;
    if(card){if(e.target.checked)card.classList.remove('unchecked');else card.classList.add('unchecked')}
    // Update the folder-level toggle text
    const section=e.target.closest('.preview-folder-section');
    if(section){
      const boxes=section.querySelectorAll('.pc-checkbox');
      const allChecked=[...boxes].every(cb=>cb.checked);
      const toggleBtn=section.querySelector('.folder-check-toggle');
      if(toggleBtn)toggleBtn.textContent=allChecked?'Uncheck All':'Check All';
    }
    updatePreviewCounts();
  }
});

// Empty folder checkboxes (delegated)
document.getElementById('empty-folder-list').addEventListener('change', function(e) {
    if (e.target.matches('[data-empty-folder]')) {
        const path = e.target.dataset.emptyFolder;
        if (e.target.checked) {
            selectedEmptyFolders.add(path);
        } else {
            selectedEmptyFolders.delete(path);
        }
        updateEmptyFolderCount();
    }
});

document.getElementById('btn-refresh-empty').addEventListener('click', loadEmptyFolders);
document.getElementById('btn-delete-empty').addEventListener('click', deleteSelectedEmptyFolders);

// ----- All Media -----
let allMediaCurrentPage = 1;
let allMediaSort = 'size_desc';

async function loadAllMedia() {
  const container = document.getElementById('all-media-container');
  container.innerHTML = '<div class="spinner"></div>';
  
  try {
    const data = await apiGet(`/api/all-files?page=${allMediaCurrentPage}&sort=${allMediaSort}&per_page=100`);
    
    document.getElementById('all-media-cur-page').textContent = data.page;
    document.getElementById('all-media-total-pages').textContent = data.total_pages;
    const pag = document.getElementById('all-media-pagination');
    pag.style.display = data.total > 0 ? 'flex' : 'none';
    document.getElementById('btn-all-media-prev').disabled = data.page <= 1;
    document.getElementById('btn-all-media-next').disabled = data.page >= data.total_pages;

    if (data.files.length === 0) {
      container.innerHTML = '<div class="empty"><p>No media files found.</p></div>';
      return;
    }
    
    let html = '';
    data.files.forEach(f => {
      const isMarked = markedAllMedia.has(f.path);
      const cls = isMarked ? 'imgbox delete-mark' : 'imgbox';
      html += `
        <div class="${cls}" data-all-path="${escapeHtml(f.path)}" data-full-path="${escapeHtml(f.path)}" data-size="${f.size}" data-type="${f.type}">
          <div class="thumb-container">
            <img src="/thumbnails/${f.thumbnail}" loading="lazy" alt="">
            ${f.type === 'video' ? '<div class="video-overlay">&#9654;</div>' : ''}
          </div>
          <div class="path" title="${escapeHtml(f.path)}">${escapeHtml(f.path)}</div>
          <div class="meta">${f.size_human} ${f.type === 'video' ? '<span class="type-badge video">VIDEO</span>' : '<span class="type-badge photo">PHOTO</span>'}</div>
        </div>`;
    });
    container.innerHTML = html;
    updateAllMediaSummary();
  } catch (e) {
    showToast("Failed to load all media: " + e.message, "error");
  }
}

const markedAllMedia = new Map();

function updateAllMediaSummary() {
  let totalSize = 0;
  markedAllMedia.forEach(s => totalSize += s);
  document.getElementById('all-media-selected-count').textContent = markedAllMedia.size;
  document.getElementById('all-media-selected-size').textContent = formatSize(totalSize);
  document.getElementById('btn-all-media-delete-marked').disabled = markedAllMedia.size === 0;
  document.getElementById('all-media-actions').style.display = 'flex';
}

async function allMediaDeleteBatch() {
  const paths = Array.from(markedAllMedia.keys());
  if (!confirm(`Move ${paths.length} file(s) to Trash?`)) return;
  
  const btn = document.getElementById('btn-all-media-delete-marked');
  btn.textContent = `Deleting ${paths.length}...`; btn.disabled = true;
  
  try {
    const resp = await apiPost('/api/delete-batch', { paths });
    markedAllMedia.clear();
    showToast(`Deleted ${resp.succeeded} file(s)`);
    await loadAllMedia();
    await refreshStats();
  } catch (e) {
    showToast("Batch delete failed: " + e.message, "error");
  } finally {
    btn.textContent = 'Delete Marked';
  }
}

// ----- Modal -----
function openFullRes(path, type) {
  const modal = document.getElementById('full-res-modal');
  const img = document.getElementById('modal-img');
  const video = document.getElementById('modal-video');
  const caption = document.getElementById('modal-caption');
  const download = document.getElementById('modal-download');
  
  caption.textContent = path;
  download.href = `/api/view-original?path=${encodeURIComponent(path)}`;
  
  if (type === 'video') {
    img.style.display = 'none';
    video.style.display = 'block';
    video.src = `/api/view-original?path=${encodeURIComponent(path)}`;
  } else {
    video.style.display = 'none';
    video.src = '';
    img.style.display = 'block';
    img.src = `/api/view-original?path=${encodeURIComponent(path)}`;
  }
  
  modal.classList.add('active');
}

function closeModal() {
  const modal = document.getElementById('full-res-modal');
  const video = document.getElementById('modal-video');
  video.pause();
  video.src = '';
  modal.classList.remove('active');
}

// ----- Small Files -----
const markedSmallFiles = new Map(); // path -> size
const keptSmallFiles = new Set();   // path -> true (persist across batches)
let smallCurrentPage = 1;

async function loadSmallFiles() {
  const sizeMb = document.getElementById('small-files-size').value || 0.5;
  const minMb = document.getElementById('small-files-min-size').value || 0.0;
  const container = document.getElementById('small-files-container');
  container.innerHTML = '<div class="spinner"></div>';
  
  try {
    const data = await apiGet(`/api/small-files?max_size_mb=${sizeMb}&min_size_mb=${minMb}&page=${smallCurrentPage}&per_page=200`);
    
    // Update pagination UI
    document.getElementById('small-cur-page').textContent = data.page;
    document.getElementById('small-total-pages').textContent = data.total_pages;
    const pag = document.getElementById('small-files-pagination');
    pag.style.display = data.total > 0 ? 'flex' : 'none';
    document.getElementById('btn-small-prev').disabled = data.page <= 1;
    document.getElementById('btn-small-next').disabled = data.page >= data.total_pages;

    if (data.files.length === 0) {
      container.innerHTML = '<div class="empty"><p>No small files found.</p></div>';
      document.getElementById('small-files-actions').style.display = 'none';
      return;
    }
    
    document.getElementById('small-files-actions').style.display = 'flex';
    const autoSelect = document.getElementById('small-files-autoselect').checked;
    let html = '';
    data.files.forEach(f => {
      // Auto-select if requested and not in kept list
      if (autoSelect && !keptSmallFiles.has(f.path)) {
        markedSmallFiles.set(f.path, f.size);
      }
      
      const isMarked = markedSmallFiles.has(f.path);
      const cls = isMarked ? 'imgbox delete-mark' : 'imgbox';
      html += `
        <div class="${cls}" data-small-path="${escapeHtml(f.path)}" data-size="${f.size}">
          <div class="thumb-container">
            <img src="/thumbnails/${f.thumbnail}" loading="lazy" alt="">
            ${f.type === 'video' ? '<div class="video-overlay">&#9654;</div>' : ''}
          </div>
          <div class="path" title="${escapeHtml(f.path)}">${escapeHtml(f.path)}</div>
          <div class="meta">${f.size_human} ${f.type === 'video' ? '<span class="type-badge video">VIDEO</span>' : '<span class="type-badge photo">PHOTO</span>'}</div>
        </div>`;
    });
    container.innerHTML = html;
    updateSmallSelectedCount();
  } catch (e) {
    showToast("Failed to load small files: " + e.message, "error");
  }
}

function updateSmallSelectedCount() {
  let totalSize = 0;
  markedSmallFiles.forEach(s => totalSize += s);
  document.getElementById('small-selected-count').textContent = markedSmallFiles.size;
  document.getElementById('small-selected-size').textContent = formatSize(totalSize);
  document.getElementById('btn-small-delete-marked').disabled = markedSmallFiles.size === 0;
  document.getElementById('small-kept-count').textContent = keptSmallFiles.size;
}

async function deleteSelectedSmallFiles() {
  const paths = Array.from(markedSmallFiles.keys());
  if (!confirm(`Are you sure you want to delete ${paths.length} files?`)) return;
  
  const btn = document.getElementById('btn-small-delete-marked');
  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Deleting...';
  
  try {
    const res = await apiPost('/api/delete-batch', { paths });
    showToast(`Successfully deleted ${res.succeeded} files.`, 'success');
    markedSmallFiles.clear();
    await loadSmallFiles();
    await refreshStats();
  } catch (e) {
    showToast("Deletion failed: " + e.message, "error");
  } finally {
    btn.textContent = oldText;
    updateSmallSelectedCount();
  }
}

document.getElementById('small-files-container').addEventListener('click', function(e) {
  const card = e.target.closest('.imgbox');
  if (!card || !card.dataset.smallPath) return;
  
  const path = card.dataset.smallPath;
  const size = parseInt(card.dataset.size);
  
  if (markedSmallFiles.has(path)) {
    markedSmallFiles.delete(path);
    keptSmallFiles.add(path); // Persist deseletion
    card.classList.remove('delete-mark');
  } else {
    markedSmallFiles.set(path, size);
    keptSmallFiles.delete(path); // Remove from kept if explicitly selected
    card.classList.add('delete-mark');
  }
  updateSmallSmallCount_v2();
});

function updateSmallSmallCount_v2() {
    // Helper to avoid duplicate code if needed, but I'll just use the existing one
    updateSmallSelectedCount();
}

document.getElementById('btn-refresh-small').addEventListener('click', () => {
    smallCurrentPage = 1;
    loadSmallFiles();
});

document.getElementById('btn-small-prev').addEventListener('click', function(){
  if(smallCurrentPage > 1){ smallCurrentPage--; loadSmallFiles(); window.scrollTo(0,0); }
});

document.getElementById('btn-small-next').addEventListener('click', function(){
  const total = parseInt(document.getElementById('small-total-pages').textContent) || 1;
  if(smallCurrentPage < total){ smallCurrentPage++; loadSmallFiles(); window.scrollTo(0,0); }
});
document.getElementById('btn-small-delete-marked').addEventListener('click', deleteSelectedSmallFiles);
document.getElementById('btn-small-reset').addEventListener('click', () => {
  markedSmallFiles.clear();
  document.querySelectorAll('#small-files-container .imgbox').forEach(el => el.classList.remove('delete-mark'));
  updateSmallSelectedCount();
});
document.getElementById('btn-small-select-all').addEventListener('click', () => {
  document.querySelectorAll('#small-files-container .imgbox').forEach(el => {
    const path = el.dataset.smallPath;
    const size = parseInt(el.dataset.size);
    markedSmallFiles.set(path, size);
    keptSmallFiles.delete(path); // If we select all, assume we want to clear from kept
    el.classList.add('delete-mark');
  });
  updateSmallSelectedCount();
});

document.getElementById('btn-small-clear-kept').addEventListener('click', () => {
    keptSmallFiles.clear();
    showToast("Kept list cleared.");
    loadSmallFiles();
});

// All Media Listeners
document.getElementById('all-media-sort').addEventListener('change', function() {
    allMediaSort = this.value;
    allMediaCurrentPage = 1;
    loadAllMedia();
});
document.getElementById('btn-refresh-all-media').addEventListener('click', loadAllMedia);
document.getElementById('btn-all-media-prev').addEventListener('click', function() {
    if(allMediaCurrentPage > 1) { 
        allMediaCurrentPage--; 
        loadAllMedia(); 
        window.scrollTo(0,0); 
    }
});
document.getElementById('btn-all-media-next').addEventListener('click', function() {
    const total = parseInt(document.getElementById('all-media-total-pages').textContent) || 1;
    if(allMediaCurrentPage < total) { 
        allMediaCurrentPage++; 
        loadAllMedia(); 
        window.scrollTo(0,0); 
    }
});
document.getElementById('btn-all-media-delete-marked').addEventListener('click', allMediaDeleteBatch);
document.getElementById('btn-all-media-reset').addEventListener('click', () => {
    markedAllMedia.clear();
    document.querySelectorAll('#all-media-container .imgbox').forEach(el => el.classList.remove('delete-mark'));
    updateAllMediaSummary();
});

document.getElementById('all-media-container').addEventListener('click', function(e) {
    const actionBtn = e.target.closest('[data-action]');
    const box = e.target.closest('.imgbox');
    if (!box) return;

    if (actionBtn) {
        const action = actionBtn.dataset.action;
        const path = box.dataset.allPath;
        const size = parseInt(box.dataset.size);

        if (action === 'all-media-mark') {
            if (markedAllMedia.has(path)) {
                markedAllMedia.delete(path);
                box.classList.remove('delete-mark');
            } else {
                markedAllMedia.set(path, size);
                box.classList.add('delete-mark');
            }
            updateAllMediaSummary();
        } else if (action === 'all-media-delete-now') {
            if (confirm(`Move this file to Trash?\n${path}`)) {
                deleteFile(path).then(() => {
                    loadAllMedia();
                });
            }
        }
        e.stopPropagation();
    }
});

document.getElementById('btn-preview-rules').addEventListener('click',previewFolderRules);

// ----- Smart Search -----
const smartSelected = new Set();
let smartSearchPollInterval = null;

async function checkSmartStatus() {
    try {
        const status = await apiGet('/api/semantic-status');
        const stText = document.getElementById('smart-status-text');
        const btnIndex = document.getElementById('btn-smart-index');
        const progContainer = document.getElementById('smart-progress-container');
        const progBar = document.getElementById('smart-progress-bar');
        const progText = document.getElementById('smart-progress-text');

        if (status.is_loading) {
            stText.textContent = "Loading Model...";
            btnIndex.disabled = true;
        } else if (status.is_indexing) {
            stText.textContent = "Indexing in progress...";
            btnIndex.disabled = true;
            progContainer.style.display = 'block';
            const pct = status.progress.total > 0 ? (status.progress.current / status.progress.total) * 100 : 0;
            progBar.style.width = pct + '%';
            progText.textContent = `${status.progress.current} / ${status.progress.total}`;
            
            if (!smartSearchPollInterval) {
                smartSearchPollInterval = setInterval(checkSmartStatus, 1000);
            }
        } else {
            stText.textContent = `Ready (${status.indexed_count} images indexed)`;
            btnIndex.disabled = false;
            progContainer.style.display = 'none';
            if (smartSearchPollInterval) {
                clearInterval(smartSearchPollInterval);
                smartSearchPollInterval = null;
            }
        }
    } catch (e) {
        console.error("Status check failed", e);
    }
}

async function triggerIndex() {
    const btn = document.getElementById('btn-smart-index');
    btn.disabled = true;
    try {
        await apiPost('/api/semantic-index', {});
        checkSmartStatus();
    } catch (e) {
        showToast("Indexing failed to start: " + e.message, "error");
        btn.disabled = false;
    }
}

async function performSmartSearch() {
    triggerSmartSearch(document.getElementById('smart-query').value.trim());
}

async function triggerSmartSearch(query, inverse=false) {
    if (!query) return;

    const btn = document.getElementById('btn-smart-search');
    const container = document.getElementById('smart-results');
    
    btn.textContent = "Searching...";
    btn.disabled = true;
    container.innerHTML = '<div class="empty"><div class="spinner"></div><p>Asking AI...</p></div>';

    try {
        const url = `/api/semantic-search?q=${encodeURIComponent(query)}&limit=100&inverse=${inverse}`;
        const data = await apiGet(url);
        renderSmartResults(data.results || [], inverse ? "Least matching: " + query : null);
    } catch (e) {
        container.innerHTML = `<div class="empty" style="color:red">Search failed: ${e.message}</div>`;
    } finally {
        btn.textContent = "Search";
        btn.disabled = false;
    }
}

function renderSmartResults(results, titleOverride=null) {
    const container = document.getElementById('smart-results');
    const actions = document.getElementById('smart-actions');
    smartSelected.clear();
    updateSmartActions();

    if (results.length === 0) {
        container.innerHTML = '<div class="empty"><p>No relevant images found.</p></div>';
        actions.style.display = 'none';
        return;
    }

    actions.style.display = 'flex';
    
    let html = '';
    if (titleOverride) {
        html += `<div style="width:100%; text-align:center; margin-bottom:10px; color:#666; font-size:13px;">${titleOverride}</div>`;
    }
    
    results.forEach(f => {
        // High confidence green border?
        // For inverse search, low score matches "not query", so score is low. 
        // We can invert confidence display or just show raw score.
        // Let's simpler: just show score.
        const score = f.score.toFixed(3);
        
        html += `<div class="imgbox" data-path="${escapeHtml(f.path)}" data-smart-path="${escapeHtml(f.path)}" data-size="${f.size}">`;
        html += `<div class="thumb-container">`;
        html += `<img src="/thumbnails/${f.thumbnail}" loading="lazy" alt="">`;
        if (f.type === 'video') html += `<div class="video-overlay">&#9654;</div>`;
        html += `</div>`;
        html += `<div class="path" title="${escapeHtml(f.path)}">Score: ${score}</div>`;
        html += `<div class="meta">${f.size_human}</div>`;
        html += `</div>`;
    });
    container.innerHTML = html;
}

window.toggleSmartSelect = function(path) {
    if (smartSelected.has(path)) {
        smartSelected.delete(path);
    } else {
        smartSelected.add(path);
    }
    
    // Update visual
    const boxes = document.querySelectorAll(`[data-smart-path="${CSS.escape(path)}"]`);
    boxes.forEach(box => {
        if (smartSelected.has(path)) box.classList.add('delete-mark');
        else box.classList.remove('delete-mark');
    });
    
    updateSmartActions();
};

function updateSmartActions() {
    const count = smartSelected.size;
    document.getElementById('smart-selected-count').textContent = count;
    document.getElementById('btn-smart-delete').disabled = count === 0;
}

async function deleteSmartSelected() {
    const paths = [...smartSelected];
    if (!paths.length) return;
    if (!confirm(`Move ${paths.length} files to Trash?`)) return;
    
    const btn = document.getElementById('btn-smart-delete');
    btn.disabled = true;
    
    const resp = await apiPost('/api/delete-batch', {paths});
    if (resp.succeeded > 0) {
        showToast(`Deleted ${resp.succeeded} files`);
        // Remove from view
        paths.forEach(p => {
             const boxes = document.querySelectorAll(`[data-smart-path="${CSS.escape(p)}"]`);
             boxes.forEach(b => b.remove());
        });
        smartSelected.clear();
        updateSmartActions();
    } else {
        showToast("Failed to delete files", "error");
    }
    btn.disabled = false;
}


// ----- Init -----
document.addEventListener('DOMContentLoaded', async () => {
    console.log("App initializing...");
    
    // Config and Path Selection
    async function refreshConfig() {
        try {
            const config = await apiGet('/api/config');
            document.getElementById('current-root-path').textContent = config.root_dir;
            document.getElementById('current-root-path').title = config.root_dir;
        } catch (e) { console.error("Config fetch failed", e); }
    }
    
    document.getElementById('btn-change-root').onclick = async () => {
        const currentPath = document.getElementById('current-root-path').textContent;
        const newPath = prompt("Enter the full folder path to scan:", currentPath);
        if (newPath && newPath !== currentPath) {
            try {
                const resp = await apiPost('/api/config', { root_dir: newPath });
                if (resp.success) {
                    showToast("Scanning new directory...");
                    await refreshConfig();
                    await pollScan();
                } else {
                    showToast(resp.error || "Failed to change directory", "error");
                }
            } catch (e) {
                showToast("Error updating directory", "error");
            }
        }
    };
    
    await refreshConfig();

    
    // Smart Search Bindings
    if(document.getElementById('btn-smart-index')) {
        document.getElementById('btn-smart-index').onclick = triggerIndex;
        document.getElementById('btn-smart-search').onclick = performSmartSearch;
        document.getElementById('btn-search-non-human').onclick = () => triggerSmartSearch("person human face", true);
        document.getElementById('smart-query').addEventListener('keyup', (e) => {
            if (e.key === 'Enter') performSmartSearch();
        });
        document.getElementById('btn-smart-select-all').onclick = () => {
           document.querySelectorAll('#smart-results .imgbox').forEach(box => {
              const path = box.dataset.smartPath;
              smartSelected.add(path);
              box.classList.add('delete-mark');
           });
           updateSmartActions();
        };
        document.getElementById('btn-smart-delete').onclick = deleteSmartSelected;
    }

    // Show "..." in stat boxes during initial load
    document.querySelectorAll('.stat-box .value').forEach(el => el.textContent = '...');
    
    try {
        const s = await apiGet('/api/stats');
        if (!s.scan_complete) {
            await pollScan();
        } else {
            document.getElementById('scan-overlay').style.display = 'none';
            await refreshAll();
            showToast('Ready to Cleanup', 'success');
        }
    } catch (e) {
        console.error("Fatal initialization error:", e);
        showToast('Connection failed. Is the server running?', 'error');
        document.getElementById('groups-container').innerHTML = `
            <div class="empty" style="color:#e74c3c">
                <div class="icon">&#9888;</div>
                <p><strong>Connection Error</strong></p>
                <p>The web UI could not communicate with the Python backend.</p>
                <button class="btn btn-primary" style="margin-top:10px" onclick="location.reload()">Retry Connection</button>
            </div>`;
    }
});

})();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(THUMB_DIR, exist_ok=True)
    
    # Pre-load index if it exists
    index_path = os.path.join(DATA_DIR, "all_files_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                all_media_files = json.load(f)
            print(f"Loaded {len(all_media_files)} files from index.")
        except Exception as e:
            print(f"Failed to load index: {e}")

    print(f"Root directory: {ROOT_DIR}")
    print(f"Thumbnails: {THUMB_DIR}")
    print(f"History file: {HISTORY_FILE}")

    # Start scan in background thread
    scan_thread = threading.Thread(target=run_scan, daemon=True)
    scan_thread.start()

    # Detect if running in Docker (don't open browser in container)
    is_docker = os.path.exists("/.dockerenv") or os.getenv("RUNNING_IN_DOCKER") == "true"
    
    # Open browser after a short delay (only if not in Docker)
    if not is_docker:
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{PORT}")
        threading.Thread(target=open_browser, daemon=True).start()

    # Bind to 0.0.0.0 in Docker for external access, 127.0.0.1 otherwise
    host = "0.0.0.0" if is_docker else "127.0.0.1"
    
    # Use production WSGI server in Docker, development server otherwise
    if is_docker:
        try:
            from waitress import serve
            print(f"Starting production server on http://{host}:{PORT}")
            serve(app, host=host, port=PORT, threads=4, channel_timeout=300)
        except ImportError:
            print("WARNING: waitress not installed, falling back to development server")
            print(f"Starting server on http://{host}:{PORT}")
            app.run(host=host, port=PORT, debug=False, threaded=True)
    else:
        print(f"Starting server on http://{host}:{PORT}")
        app.run(host=host, port=PORT, debug=False, threaded=True)

