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

ROOT_DIR = os.path.expanduser("~/Pictures") # Default directory to scan
PORT = 8080
THUMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbnails")
THUMB_SIZE = (150, 150)
THUMB_QUALITY = 50
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deletion_history.json")
PARTIAL_HASH_SIZE = 8192  # 8 KB for partial hashing

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
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"total_space_saved": 0, "deletions": []}


def enrich_history_from_index():
    """Use all_files_index.json to find 'kept' versions for old history entries."""
    global deletion_history
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_files_index.json")
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
    all_files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_files_index.json")
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

# ---------------------------------------------------------------------------
# Trash helpers
# ---------------------------------------------------------------------------

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
        self.embeddings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_embeddings.json")
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
        all_files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_files_index.json")
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
<title>Duplicate Photo &amp; Video Finder</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f0f2f5;color:#333;padding:20px;min-height:100vh}
h1{font-size:24px;margin-bottom:20px;color:#1a1a2e}
a{color:#4361ee}

/* Stats */
.stats{display:flex;gap:14px;margin-bottom:20px;flex-wrap:wrap}
.stat-box{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:18px 22px;border-radius:12px;min-width:170px;flex:1}
.stat-box h3{font-size:12px;opacity:.85;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}
.stat-box .value{font-size:26px;font-weight:700}
.stat-box.green{background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%)}

/* Tabs */
.tabs{display:flex;gap:4px;margin-bottom:0}
.tab{padding:10px 22px;background:#ddd;border:none;cursor:pointer;border-radius:8px 8px 0 0;font-size:14px;transition:background .15s}
.tab:hover{background:#ccc}
.tab.active{background:#fff;font-weight:600}
.tab-content{display:none;background:#fff;padding:20px;border-radius:0 8px 8px 8px}
.tab-content.active{display:block}

/* Action bar */
.action-bar{position:sticky;top:0;z-index:100;background:#fff;padding:14px 18px;border-radius:10px;margin-bottom:18px;box-shadow:0 2px 12px rgba(0,0,0,.08);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}
.action-bar .summary{font-size:14px;color:#555}
.action-bar .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}

/* Buttons */
.btn{padding:7px 14px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;transition:opacity .15s}
.btn:hover{opacity:.85}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-keep{background:#27ae60;color:#fff}
.btn-delete{background:#e74c3c;color:#fff}
.btn-primary{background:#4361ee;color:#fff}
.btn-danger{background:#e74c3c;color:#fff}
.btn-secondary{background:#6c757d;color:#fff}
.btn-success{background:#27ae60;color:#fff}
.btn-lg{padding:10px 20px;font-size:14px}
select.sort-select{padding:7px 10px;font-size:13px;border-radius:6px;border:1px solid #ccc}

/* Groups */
.group{margin-bottom:18px;background:#fff;padding:16px;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.06);transition:opacity .3s}
.group-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.group-header h3{font-size:14px;color:#333}
.images{display:flex;gap:12px;flex-wrap:wrap}
.imgbox{width:190px;border:2px solid transparent;padding:10px;border-radius:8px;transition:all .2s}
.imgbox.keep{border-color:#27ae60;background:#e8f5e9}
.imgbox.delete-mark{border-color:#e74c3c;background:#ffebee;opacity:.75}
.imgbox .thumb-container{position:relative;min-height:80px;display:flex;align-items:center;justify-content:center;background:#f5f5f5;border-radius:6px;overflow:hidden}
.imgbox img{max-width:170px;max-height:120px;border-radius:4px;cursor:pointer;display:block}
.imgbox .video-overlay{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:36px;color:#fff;text-shadow:0 0 12px rgba(0,0,0,.7);pointer-events:none}
.imgbox .path{font-size:9px;word-break:break-all;margin-top:6px;color:#777;max-height:34px;overflow:hidden}
.imgbox .meta{font-size:10px;color:#999;margin-top:2px}
.type-badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:9px;font-weight:700;margin-left:4px}
.type-badge.video{background:#9b59b6;color:#fff}
.type-badge.photo{background:#3498db;color:#fff}
.imgbox .actions{margin-top:8px;display:flex;gap:6px}

    /* Group Collapse/Expand */
    .group-header { cursor: pointer; user-select: none; }
    .group-header:hover { background-color: #f9f9f9; }
    .group-toggle-icon { display: inline-block; margin-right: 8px; transition: transform 0.2s; }
    .group-body { display: block; margin-top: 12px; } /* Expanded by default */
    .group.collapsed .group-toggle-icon { transform: rotate(-90deg); }

    /* Kept Info in History */
    .kept-info { margin-top: 4px; padding: 4px 8px; background: #e8f5e9; border-radius: 4px; font-size: 11px; color: #2e7d32; border: 1px solid #c8e6c9; display: inline-block; }
    .kept-info strong { font-weight: 600; }

/* Pagination */
.pagination{display:flex;justify-content:center;align-items:center;gap:12px;margin:20px 0;padding:14px;background:#fff;border-radius:10px}
.pagination button{padding:8px 18px;border:1px solid #ddd;background:#fff;cursor:pointer;border-radius:6px;font-size:13px}
.pagination button:hover:not(:disabled){background:#f0f0f0}
.pagination button:disabled{opacity:.4;cursor:not-allowed}
.page-info{font-size:13px;color:#666}

/* Folder list */
.folder-list{max-height:500px;overflow-y:auto}
.folder-item{padding:10px 12px;border-bottom:1px solid #eee;display:flex;align-items:center;gap:10px}
.folder-item:hover{background:#f9f9f9}
.folder-item input[type="checkbox"]{width:18px;height:18px;cursor:pointer;flex-shrink:0}
.folder-item.kept{background:#e8f5e9;border-left:4px solid #2e7d32}
.folder-item.to-delete{background:#ffebee;border-left:4px solid #c62828}
.folder-path{font-family:monospace;font-size:12px;word-break:break-all;flex:1}
.folder-count{background:#7f8c8d;color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;margin-left:10px;white-space:nowrap}
.folder-count.delete-badge{background:#e74c3c}
.folder-count.keep-badge{background:#27ae60}
.folder-actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:16px;padding:14px;background:#fff;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.08)}
.folder-actions .info{font-size:13px;color:#555;flex:1}
.preview-box{margin:16px 0;padding:16px;background:#fffde7;border:1px solid #ffe082;border-radius:8px;font-size:13px}
.preview-box strong{color:#e65100}

/* History */
.history-folder-section{margin-bottom:20px}
.history-folder-header{display:flex;align-items:center;gap:8px;padding:10px 12px;background:#f0f2f5;border-radius:8px 8px 0 0;font-size:12px;font-weight:600;color:#333;border:1px solid #e0e0e0;border-bottom:none;cursor:pointer;user-select:none;transition:background .15s}
.history-folder-header:hover{background:#e4e6eb}
.history-folder-header .hist-folder-toggle{font-size:12px;color:#666;flex-shrink:0;transition:transform .2s}
.history-folder-header .hist-folder-toggle.open{transform:rotate(90deg)}
.history-folder-header .hist-folder-label{flex:1;word-break:break-all}
.history-folder-body{border:1px solid #e0e0e0;border-radius:0 0 8px 8px;overflow:hidden}
.history-folder-body.collapsed{display:none}
.history-item{padding:10px 12px;border-bottom:1px solid #eee;display:flex;justify-content:space-between;font-size:13px}
.history-item:last-child{border-bottom:none}
.history-item .hist-path{font-family:monospace;font-size:11px;word-break:break-all;flex:1;color:#555}
.history-item .hist-meta{text-align:right;white-space:nowrap;margin-left:12px;color:#888;font-size:11px}

/* Toast */
.toast-container{position:fixed;top:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px}
.toast{padding:12px 20px;border-radius:8px;color:#fff;font-size:13px;box-shadow:0 4px 14px rgba(0,0,0,.15);animation:toastIn .3s ease;max-width:380px}
.toast.success{background:#27ae60}
.toast.error{background:#e74c3c}
.toast.info{background:#4361ee}
@keyframes toastIn{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}

/* Scan overlay */
.scan-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.6);z-index:10000;display:flex;align-items:center;justify-content:center}
.scan-box{background:#fff;padding:40px 50px;border-radius:16px;text-align:center;max-width:420px}
.scan-box h2{margin-bottom:12px}
.scan-box .phase{color:#666;font-size:14px;margin-top:8px}
.spinner{width:40px;height:40px;border:4px solid #e0e0e0;border-top-color:#4361ee;border-radius:50%;animation:spin .8s linear infinite;margin:16px auto}
@keyframes spin{to{transform:rotate(360deg)}}

/* Empty state */
.empty{text-align:center;padding:60px 20px;color:#999}
.empty .icon{font-size:48px;margin-bottom:12px}

/* Folder-rule visual preview */
.preview-summary-bar{padding:14px 18px;background:#fff;border-radius:10px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,.08);font-size:14px;color:#555}
.preview-summary-bar strong{color:#333}
.preview-folder-section{background:#fff;border-radius:10px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.06);overflow:hidden}
.preview-folder-header{display:flex;align-items:center;gap:10px;padding:12px 16px;cursor:pointer;user-select:none;background:#fafafa;border-bottom:1px solid #eee;transition:background .15s}
.preview-folder-header:hover{background:#f0f0f0}
.preview-folder-header .folder-toggle{font-size:12px;color:#888;transition:transform .2s;flex-shrink:0}
.preview-folder-header .folder-toggle.open{transform:rotate(90deg)}
.preview-folder-header .folder-name{font-family:monospace;font-size:12px;word-break:break-all;flex:1;color:#333}
.preview-folder-header .folder-badge{background:#e74c3c;color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;white-space:nowrap}
.preview-folder-header .folder-check-toggle{padding:4px 10px;border:1px solid #ccc;border-radius:5px;font-size:11px;cursor:pointer;background:#fff;white-space:nowrap}
.preview-folder-header .folder-check-toggle:hover{background:#f0f0f0}
.preview-folder-body{padding:12px 16px}
.preview-folder-body.collapsed{display:none}
.preview-grid{display:flex;flex-wrap:wrap;gap:12px}
.preview-card{width:170px;border:2px solid #e74c3c;padding:8px;border-radius:8px;background:#ffebee;transition:all .2s;position:relative}
.preview-card.unchecked{border-color:#ccc;background:#f9f9f9;opacity:.6}
.preview-card .pc-checkbox{position:absolute;top:6px;left:6px;z-index:2;width:18px;height:18px;cursor:pointer;accent-color:#e74c3c}
.preview-card .pc-thumb{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:4px;display:block;background:#f5f5f5}
.preview-card .pc-name{font-size:10px;word-break:break-all;margin-top:5px;color:#555;max-height:28px;overflow:hidden;line-height:1.3}
.preview-card .pc-meta{font-size:10px;color:#999;margin-top:2px}
.preview-card .video-badge{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:28px;color:#fff;text-shadow:0 0 10px rgba(0,0,0,.6);pointer-events:none}
.preview-bottom-bar{position:fixed;bottom:0;left:0;right:0;z-index:200;background:#fff;border-top:2px solid #e0e0e0;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 -4px 16px rgba(0,0,0,.1)}
.preview-bottom-bar .pbb-info{font-size:14px;color:#555}
.preview-bottom-bar .pbb-info strong{color:#333}
.preview-bottom-bar .pbb-actions{display:flex;gap:10px}

/* Modal */
.modal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.9);z-index:11000;align-items:center;justify-content:center;flex-direction:column}
.modal.active{display:flex}
.modal-close{position:absolute;top:20px;right:30px;color:#fff;font-size:40px;cursor:pointer;user-select:none}
.modal-content{max-width:95%;max-height:85%;object-fit:contain;box-shadow:0 0 30px rgba(0,0,0,.5)}
.modal-caption{color:#fff;margin-top:20px;font-size:14px;text-align:center;word-break:break-all;max-width:80%}
.modal-actions{margin-top:15px;display:flex;gap:15px}
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

<h1>Duplicate Photo &amp; Video Finder</h1>

<div class="stats" id="stats-bar">
  <div class="stat-box"><h3>Duplicate Groups</h3><div class="value" id="stat-groups">--</div></div>
  <div class="stat-box"><h3>Duplicate Photos</h3><div class="value" id="stat-photos">--</div></div>
  <div class="stat-box"><h3>Duplicate Videos</h3><div class="value" id="stat-videos">--</div></div>
  <div class="stat-box"><h3>Wasted Space</h3><div class="value" id="stat-wasted">--</div></div>
  <div class="stat-box green"><h3>Space Saved</h3><div class="value" id="stat-saved">--</div></div>
</div>

<div class="tabs">
  <button class="tab active" data-tab="visual">Visual View</button>
  <button class="tab" data-tab="folders">By Folder</button>
  <button class="tab" data-tab="all-media">All Media</button>
  <button class="tab" data-tab="empty-folders">Empty Folders</button>
  <button class="tab" data-tab="small-files">Small Files / Screenshot Cleanup</button>
  <button class="tab" data-tab="history">Deleted Files</button>
  <button class="tab" data-tab="smart-search" style="background:#e0f7fa; color:#006064;">&#10024; Smart Search</button>
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
  
  <div id="all-media-container" class="images" style="margin-top:20px; display:flex; gap:12px; flex-wrap:wrap"></div>

  <div class="pagination" id="all-media-pagination" style="display:none; margin-top:20px">
    <button id="btn-all-media-prev">&laquo; Previous</button>
    <span class="page-info">Page <span id="all-media-cur-page">1</span> of <span id="all-media-total-pages">1</span></span>
    <button id="btn-all-media-next">Next &raquo;</button>
  </div>
</div>

<!-- Empty Folders -->
<div id="empty-folders" class="tab-content">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
    <div>
      <h2 style="margin-bottom:2px">Empty Folders</h2>
      <p style="color:#888;font-size:13px">Folders that contain no files (ignoring .DS_Store).</p>
    </div>
    <button class="btn btn-primary" id="btn-refresh-empty">Refresh List</button>
  </div>
  
  <div class="folder-actions" id="empty-folder-actions" style="display:none">
    <div class="info"><span id="empty-folder-selected-count">0</span> folder(s) selected</div>
    <button class="btn btn-lg btn-danger" id="btn-delete-empty" disabled>Delete Selected Folders</button>
  </div>
  
  <div id="empty-folder-list" class="folder-list"></div>
</div>

<!-- Small Files -->
<div id="small-files" class="tab-content">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap: wrap; gap: 10px;">
    <div>
      <h2 style="margin-bottom:2px">Small Files / Screenshots</h2>
      <p style="color:#888;font-size:13px">Find all images and videos smaller than a specific size.</p>
    </div>
    <div style="display:flex; align-items:center; gap:8px">
        <label style="font-size:13px">Min (MB):</label>
        <input type="number" id="small-files-min-size" value="0.0" step="0.1" min="0" style="width:70px; padding:6px; border:1px solid #ccc; border-radius:6px">
        
        <label style="font-size:13px">Max (MB):</label>
        <input type="number" id="small-files-size" value="0.5" step="0.1" min="0" style="width:70px; padding:6px; border:1px solid #ccc; border-radius:6px">
        <button class="btn btn-primary" id="btn-refresh-small">Search</button>
    </div>
  </div>
  
  <div class="action-bar" id="small-files-actions" style="display:none">
    <div class="summary">
      <span id="small-selected-count">0</span> files marked for deletion
      (<span id="small-selected-size">0 B</span>)
    </div>
    <div class="controls">
      <label style="font-size:12px; cursor:pointer; margin-right:8px"><input type="checkbox" id="small-files-autoselect" checked> Auto-select (skips Kept)</label>
      <button class="btn btn-secondary" id="btn-small-clear-kept">Clear Kept List (<span id="small-kept-count">0</span>)</button>
      <button class="btn btn-primary" id="btn-small-select-all">Select All</button>
      <button class="btn btn-danger" id="btn-small-delete-marked" disabled>Delete Marked</button>
      <button class="btn btn-secondary" id="btn-small-reset">Reset</button>
    </div>
  </div>
  
  <div id="small-files-container" class="images" style="margin-top:20px; display:flex; gap:12px; flex-wrap:wrap"></div>

  <div class="pagination" id="small-files-pagination" style="display:none; margin-top:20px">
    <button id="btn-small-prev">&laquo; Previous</button>
    <span class="page-info">Page <span id="small-cur-page">1</span> of <span id="small-total-pages">1</span></span>
    <button id="btn-small-next">Next &raquo;</button>
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

  <div id="smart-results" class="images" style="display:flex; gap:12px; flex-wrap:wrap; justify-content:center;"></div>
</div>

<!-- History -->
<div id="history" class="tab-content">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
    <div>
      <h2 style="margin-bottom:2px">Deleted Files</h2>
      <p style="color:#888;font-size:13px">Files moved to Trash. Total space recovered: <strong id="hist-total">0 B</strong></p>
    </div>
    <button class="btn btn-secondary" onclick="loadHistory()">Refresh &amp; Re-link</button>
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
      document.getElementById('stat-photos').textContent = s.total_photos ?? 0;
      document.getElementById('stat-videos').textContent = s.total_videos ?? 0;
      document.getElementById('stat-wasted').textContent = s.wasted_space_human ?? '0 B';
      document.getElementById('stat-saved').textContent = s.space_saved_human ?? '0 B';
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
async function loadHistory(){
  const el=document.getElementById('history-list');
  el.innerHTML = '<div class="empty"><div class="spinner"></div><p>Calculating file locations...</p></div>';
  
  try {
      const data = await apiGet('/api/history?enrich=true');
      document.getElementById('hist-total').textContent = formatSize(data.total_space_saved);
      
      if(!data.deletions || !data.deletions.length){
          el.innerHTML = '<div class="empty"><p>No files deleted yet.</p></div>';
          return;
      }
      
      const items = [...data.deletions].reverse();
      const byFolder = new Map();
      items.forEach(d => {
          const folder = getFolderFromPath(d.path);
          if(!byFolder.has(folder)) byFolder.set(folder, []);
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
              const ts = dt.toLocaleDateString() + ' ' + dt.toLocaleTimeString();
              const fname = d.path ? d.path.split(/[/\\]/).pop() : 'Unknown';
              
              let keptHtml = '';
              if (d.kept_versions && d.kept_versions.length > 0) {
                  // If we have multiple, show first and a count
                  const keptPath = d.kept_versions[0];
                  const keptName = keptPath.split(/[/\\]/).pop();
                  const keptFolder = getFolderFromPath(keptPath);
                  const moreCount = d.kept_versions.length - 1;
                  const moreText = moreCount > 0 ? ` (+${moreCount} other locations)` : '';
                  
                  keptHtml = `<div class="kept-info">
                      <strong>Kept Original:</strong> ${escapeHtml(keptName)}${moreText}<br>
                      <small style="opacity:0.8">Location: ${escapeHtml(keptFolder)}</small>
                  </div>`;
              }
                
              return `<div class="history-item">
                <div style="flex:1">
                    <span class="hist-path" title="${escapeHtml(d.path)}">${escapeHtml(fname)}</span>
                    ${keptHtml}
                </div>
                <span class="hist-meta">${d.size_human||''}<br>${ts}</span>
              </div>`;
          }).join('');
          
          return `<div class="history-folder-section" data-hist-section="${sectionId}">
              <div class="history-folder-header" data-hist-toggle="${sectionId}">
                  <span class="hist-folder-toggle">&#9654;</span>
                  <span class="hist-folder-label">${escapeHtml(folder)} &middot; ${fileCount} file(s) &middot; ${sizeHuman}</span>
              </div>
              <div class="history-folder-body collapsed" data-hist-body="${sectionId}">
                  ${body}
              </div>
          </div>`;
      }).join('');
  } catch (e) {
      console.error("History load failed:", e);
      el.innerHTML = `<div class="empty" style="color:#e74c3c"><p>Failed to load deletion history.</p></div>`;
  }
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

// ----- Event delegation -----
document.addEventListener('click',function(e){
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
    if(id === 'empty-folders') loadEmptyFolders();
    if(id === 'all-media') loadAllMedia();
    if(id === 'small-files') loadSmallFiles();
    if(id === 'smart-search') checkSmartStatus();
    return;
  }

  // Modal close
  if(e.target.matches('.modal-close') || e.target.matches('.modal')){
    closeModal();
    return;
  }

  // Open Full Res from ANY imgbox
  const imgInBox = e.target.closest('.imgbox img');
  if(imgInBox){
    const box = imgInBox.closest('.imgbox');
    // Try different data attributes used in different tabs
    const path = box.dataset.fullPath || box.dataset.path || box.dataset.smallPath;
    const type = box.dataset.type || (box.querySelector('.type-badge.video') ? 'video' : 'photo');
    if(path) {
        openFullRes(path, type);
        return;
    }
  }

  // 3. Group Actions (Keep, Delete, Reset)
  const box = e.target.closest('.imgbox');
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
          <div class="actions" style="margin-top:8px; display:flex; gap:4px">
            <button class="btn btn-delete btn-action" data-action="all-media-mark" style="flex:1; padding:4px; font-size:11px">Mark</button>
            <button class="btn btn-danger btn-action" data-action="all-media-delete-now" style="flex:1; padding:4px; font-size:11px">Delete</button>
          </div>
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
        
        html += `<div class="imgbox" data-path="${escapeHtml(f.path)}" data-smart-path="${escapeHtml(f.path)}">`;
        html += `<div class="thumb-container" onclick="toggleSmartSelect('${escapeHtml(f.path).replace(/'/g, "\\'")}')">`;
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
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_files_index.json")
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

    # Open browser after a short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"Starting server on http://localhost:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
