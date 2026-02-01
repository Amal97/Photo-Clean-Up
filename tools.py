#!/usr/bin/env python3
import sys
import os
import json
import time
from datetime import datetime

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Import existing logic from clean.py
from clean import (
    SemanticSearch, ROOT_DIR, ALL_MEDIA_EXTS, PHOTO_EXTS,
    duplicates, all_media_files, move_to_trash,
    remove_file_from_duplicates, remove_file_from_index,
    save_history, deletion_history, format_size
)

def run_indexing():
    print(f"\n[Semantic Indexing] Root: {ROOT_DIR}")
    search = SemanticSearch()
    
    # Load model in main process for direct execution
    print("Loading CLIP model... (this may take a moment)")
    search.load_model()
    
    # Determine targets
    targets = []
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_files_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                data = json.load(f)
                targets = [f["path"] for f in data if os.path.splitext(f["path"])[1].lower() in PHOTO_EXTS]
            print(f"Loaded {len(targets)} photos from all_files_index.json")
        except:
            pass
            
    if not targets:
        print("No index found. Performing quick directory walk...")
        for dirpath, _, filenames in os.walk(ROOT_DIR):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in PHOTO_EXTS:
                    targets.append(os.path.join(dirpath, name))
        print(f"Found {len(targets)} files via walk.")

    # Filter out already indexed
    to_index = [p for p in targets if p not in search.embeddings]
    print(f"Total: {len(targets)} | Already indexed: {len(search.embeddings)} | Remaining: {len(to_index)}")
    
    if not to_index:
        print("Everything is already indexed!")
        return

    print("Starting indexing loop. Press Ctrl+C to stop (it will auto-save).")
    try:
        search._index_loop(to_index)
    except KeyboardInterrupt:
        print("\nStopping and saving progress...")
    finally:
        search.save_embeddings()
        print(f"Embeddings saved to {len(search.embeddings)} entries.")

def run_cleanup_rules(keep_folder_prefix):
    """Example of a direct cleanup script based on a folder rule."""
    print(f"\n[Cleanup] Identifying duplicates outside of: {keep_folder_prefix}")
    # Note: run_scan() needs to be run or duplicates loaded
    from clean import run_scan
    
    print("Scanning for duplicates first...")
    run_scan() # This populates 'duplicates' global
    
    to_delete = []
    for h, files in duplicates.items():
        kept = [f for f in files if f["path"].startswith(keep_folder_prefix)]
        others = [f for f in files if not f["path"].startswith(keep_folder_prefix)]
        
        if kept and others:
            to_delete.extend(others)

    if not to_delete:
        print("No duplicates found to delete based on that rule.")
        return

    total_size = sum(f["size"] for f in to_delete)
    print(f"Found {len(to_delete)} files to delete ({format_size(total_size)}).")
    confirm = input("Proceed with deletion? (y/n): ")
    
    if confirm.lower() == 'y':
        for f in to_delete:
            path = f["path"]
            size = f["size"]
            ok, err = move_to_trash(path)
            if ok:
                remove_file_from_duplicates(path)
                remove_file_from_index(path)
                deletion_history["total_space_saved"] += size
                deletion_history["deletions"].append({
                    "path": path, "size": size, "timestamp": datetime.now().isoformat()
                })
                print(f"Deleted: {os.path.basename(path)}")
            else:
                print(f"Failed: {path} - {err}")
        save_history(deletion_history)
        print("Cleanup complete.")
    else:
        print("Aborted.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tools.py index          # Run semantic indexing")
        print("  python tools.py cleanup [path] # Delete duplicates NOT in [path]")
        sys.exit(1)
        
    cmd = sys.argv[1]
    if cmd == "index":
        run_indexing()
    elif cmd == "cleanup" and len(sys.argv) > 2:
        run_cleanup_rules(sys.argv[2])
    else:
        print("Invalid command or missing arguments.")
