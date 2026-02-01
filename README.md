# 📸 Photo-Clean-Up

A powerful tool for managing and cleaning up large photo and video libraries. Features a modern web interface with AI-powered duplicate detection and semantic search.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

## ✨ Features

- **🚀 Smart Duplicate Detection**: 3-phase scanning (size → partial hash → full SHA-256) for 100% accuracy
- **🧠 AI-Powered Search**: Find photos using natural language (e.g., "sunset on beach", "family dinner") via CLIP
- **📁 Folder Cleanup**: Identify and remove duplicates across folders
- **🗑️ Safe Deletion**: Moves files to trash instead of permanent deletion
- **📂 Empty Folder Finder**: Automatically detect and remove empty directories
- **🎥 Video Support**: Duplicate detection and thumbnail generation with FFmpeg
- **📱 Modern Web UI**: Clean, responsive interface for managing your entire library

## 🚀 Quick Start (Docker - Recommended)

**3 steps to get started:**

```bash
# 1. Clone and navigate
git clone https://github.com/Amal97/Photo-Clean-Up.git
cd Photo-Clean-Up

# 2. Configure your photos directory
cp .env.example .env
# Edit .env: PHOTOS_DIR=/path/to/your/photos

# 3. Launch
docker-compose up -d
```

**Access at:** `http://localhost:8080`

**View logs:** `docker-compose logs -f`

### Important: Accessing the Container
✅ Always use `http://localhost:8080` on your host machine  
❌ Don't try to access the container's internal IP (e.g., 172.x.x.x)

The container can only scan directories you mount. To scan a different directory:
```bash
# Edit .env file
PHOTOS_DIR=/path/to/different/photos

# Restart
docker-compose down
docker-compose up -d
```

### Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PHOTOS_DIR` | - | **Required**: Path to your photos on host machine |
| `PORT` | `8080` | Web server port |
| `IMMICH_API_URL` | - | Optional: Immich integration endpoint |
| `IMMICH_API_KEY` | - | Optional: Immich API key |

**Advanced setup?** See [DOCKER.md](DOCKER.md) for GPU support, reverse proxies, Kubernetes, and more.

---

## 🛠️ Local Installation (Without Docker)

### Prerequisites
- Python 3.8+
- FFmpeg (for video thumbnails)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### Setup

```bash
# Clone repository
git clone https://github.com/Amal97/Photo-Clean-Up.git
cd Photo-Clean-Up

# Install dependencies
pip install -r requirements.txt

# Run
python clean.py
```

**Access at:** `http://localhost:8080`

### Configuration

You can set the photos directory via:
1. **Environment variable** (recommended):
   ```bash
   export ROOT_DIR=/path/to/photos
   python clean.py
   ```

2. **Settings file**: Edit via the web UI Settings tab

3. **Code**: Edit `ROOT_DIR` in `clean.py` (line 38)

---

## 📖 Usage

### Web Interface
- **Duplicates Tab**: Review and delete duplicate files
- **Folders Tab**: Find folders containing duplicates
- **Smart Search**: AI-powered natural language search
- **Small Files**: Find and remove screenshots/small images
- **Empty Folders**: Detect and clean empty directories
- **All Media**: Browse and manage your entire library
- **History**: Track deleted files and space saved

### CLI Tools

```bash
# Index images for Smart Search
python tools.py index

# Cleanup based on folder rules
python tools.py cleanup /path/to/keep/folder
```

---

## 🧠 Smart Search (CLIP)

First run downloads the CLIP model (~600MB) from Hugging Face.

- **macOS**: Automatically uses Apple Silicon (MPS) for faster indexing
- **Linux/Windows**: Uses CPU (or CUDA if available)
- **Performance**: ~200-2000 images/minute depending on hardware

---

## 🐳 Docker Tips

**View logs:**
```bash
docker-compose logs -f
```

**Stop:**
```bash
docker-compose down
```

**Rebuild after code changes:**
```bash
docker-compose up -d --build
```

**Permission errors?**
```bash
# Edit docker-compose.yml, add:
user: "1000:1000"  # Use your host user ID
```

**Port conflict?**
```bash
# Edit .env file:
PORT=8090  # Use different port
```

---

## ⚖️ License & Attribution

This project is licensed under the **MIT License**.

### Attribution Requirement
If you use this software in your projects (open-source or commercial), you must:
1. Retain the `LICENSE` and `NOTICE` files
2. Link back to this repository: `https://github.com/Amal97/Photo-Clean-Up`
3. Credit **Amal Chandra** in your documentation

---

*Built with ❤️ by [Amal Chandra](https://github.com/Amal97)*
