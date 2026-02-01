# 📸 Photo-Clean-Up

A comprehensive tool for managing and cleaning up large photo and video libraries. It features a modern web interface to find duplicates, remove empty folders, and perform semantic "Smart Search" using AI (CLIP).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-orange)

## ✨ Features

- **🚀 3-Phase Duplicate Detection**: Efficiently scans for duplicate media using a graduated approach:
  1. File size grouping.
  2. Partial hashing (header/footer).
  3. Full SHA-256 hashing for 100% accuracy.
- **🧠 Smart Search (AI-Powered)**: Search your photo library using natural language (e.g., "sunset on the beach", "cute cat", "family dinner") powered by OpenAI's CLIP model.
- **📁 Folder-Level Cleanup**: Identify and clean up specific folders that contain duplicates that are already saved elsewhere.
- **🗑️ Trash Integration**: Safely moves files to the system trash instead of permanent deletion.
- **📂 Empty Folder Finder**: Quickly find and remove empty or "effectively empty" folders (containing only `.DS_Store`).
- **📱 Responsive Web UI**: A clean, interactive dashboard to manage your entire media library.
- **🎥 Video Support**: Identifies duplicate videos and generates video thumbnails using FFmpeg.

## 🛠️ Installation

### 1. Prerequisites
- **Python 3.8+**
- **FFmpeg** (for video thumbnails)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/Photo-Clean-Up.git
cd Photo-Clean-Up
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Open `clean.py` and update the `ROOT_DIR` variable to point to your photos directory:
```python
ROOT_DIR = "/path/to/your/photos"
```

## 🚀 Usage

### Start the Web Interface
```bash
python clean.py
```
After starting, open your browser and navigate to `http://localhost:8080`.

### CLI Tools
You can also use `tools.py` for headless operations:

- **Index images for Smart Search**:
  ```bash
  python tools.py index
  ```
- **Cleanup based on folder rules**:
  ```bash
  python tools.py cleanup /path/to/keep/folder
  ```

## 🧠 Smart Search (CLIP)

The Smart Search feature utilizes the `openai/clip-vit-base-patch32` model.
- **First Run**: It will download the model (~600MB) from Hugging Face.
- **Acceleration**: On macOS, it automatically uses **Apple Silicon (MPS)** for faster indexing.
- **Indexing**: Images are indexed in batches for efficiency. Once indexed, search is instantaneous.

## 📝 File Structure

- `clean.py`: Main Flask application and business logic.
- `tools.py`: CLI utilities for indexing and rules-based cleanup.
- `requirements.txt`: Python dependencies.
- `all_files_index.json`: Cache of scanned files.
- `image_embeddings.json`: AI-generated embeddings for Smart Search.
- `deletion_history.json`: Tracks total space saved and deleted files.
- `thumbnails/`: Cache for UI thumbnails.

## ⚖️ License & Attribution

This project is licensed under the **MIT License**. 

### Special Attribution Requirement
If you use this software, or portions of it, in your own projects (whether open-source or commercial), **you must provide attribution** by:
1. Retaining the original `LICENSE` and `NOTICE` files in your repository.
2. Including a link back to this original repository: `https://github.com/Amal97/Photo-Clean-Up`
3. Crediting **Amal Chandra** in your project's documentation or "About" section.

---
*Built with ❤️ by [Amal Chandra](https://github.com/Amal97)*
