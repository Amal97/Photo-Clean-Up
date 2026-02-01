# 🐳 Docker Deployment Guide

Advanced Docker deployment guide for Photo-Clean-Up.

## Table of Contents
- [Building Custom Images](#building-custom-images)
- [Multi-Architecture Support](#multi-architecture-support)
- [GPU Acceleration](#gpu-acceleration)
- [Reverse Proxy Setup](#reverse-proxy-setup)
- [Performance Tuning](#performance-tuning)
- [Kubernetes Deployment](#kubernetes-deployment)

## Building Custom Images

### Basic Build
```bash
docker build -t photo-cleanup:latest .
```

### Build with Custom Arguments
```bash
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t photo-cleanup:custom \
  .
```

### Multi-Stage Build Benefits
The included Dockerfile uses a single-stage build optimized for:
- **Small image size**: ~800MB-1.5GB (depending on platform)
- **Security**: Non-root user (`appuser`)
- **Production-ready**: Includes health checks and proper signal handling

## Multi-Architecture Support

Build for multiple architectures (useful for Raspberry Pi, ARM servers, etc.):

```bash
# Enable buildx (if not already enabled)
docker buildx create --name multiarch --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t yourusername/photo-cleanup:latest \
  --push \
  .
```

Supported platforms:
- `linux/amd64` - Standard x86_64 (Intel/AMD)
- `linux/arm64` - ARM 64-bit (Apple Silicon, Raspberry Pi 4+, AWS Graviton)

## GPU Acceleration

### NVIDIA GPU Support (Linux Only)

The CLIP model can leverage NVIDIA GPUs for faster indexing on Linux systems.

**Prerequisites:**
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed

**Modified docker-compose.yml:**
```yaml
services:
  photo-cleanup:
    # ... other config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Manual Docker Run:**
```bash
docker run -d \
  --name photo-cleanup \
  --gpus all \
  -p 8080:8080 \
  -v ~/Pictures:/photos:ro \
  -v photo-cleanup-data:/app/data \
  photo-cleanup:latest
```

### Performance Notes
- **CPU Mode**: ~200-500 images/minute (typical)
- **Apple Silicon (MPS)**: ~1000-2000 images/minute (native only, not in Docker)
- **NVIDIA GPU (CUDA)**: ~2000-5000 images/minute (Linux Docker with GPU)

## Reverse Proxy Setup

### Nginx

```nginx
server {
    listen 80;
    server_name photos.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for long-running operations
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
```

### Traefik (docker-compose labels)

```yaml
services:
  photo-cleanup:
    # ... other config ...
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.photo-cleanup.rule=Host(`photos.example.com`)"
      - "traefik.http.routers.photo-cleanup.entrypoints=websecure"
      - "traefik.http.routers.photo-cleanup.tls.certresolver=letsencrypt"
      - "traefik.http.services.photo-cleanup.loadbalancer.server.port=8080"
```

### Caddy

```caddy
photos.example.com {
    reverse_proxy localhost:8080
    
    # Increase timeouts for long operations
    transport http {
        read_timeout 5m
        write_timeout 5m
    }
}
```

## Performance Tuning

### Memory Allocation

For large photo libraries (100k+ images), increase memory limits:

```yaml
services:
  photo-cleanup:
    deploy:
      resources:
        limits:
          memory: 8G  # Increase from default 4G
        reservations:
          memory: 2G
```

### CPU Allocation

For faster scanning and indexing:

```yaml
services:
  photo-cleanup:
    deploy:
      resources:
        limits:
          cpus: '4'  # Increase from default 2
```

### Volume Performance

For better thumbnail performance, use a tmpfs mount (RAM-based):

```yaml
services:
  photo-cleanup:
    tmpfs:
      - /app/data/thumbnails:size=2G,mode=1755
```

**Note**: Thumbnails will be regenerated on restart with this approach.

## Kubernetes Deployment

### Basic Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: photo-cleanup
spec:
  replicas: 1
  selector:
    matchLabels:
      app: photo-cleanup
  template:
    metadata:
      labels:
        app: photo-cleanup
    spec:
      containers:
      - name: photo-cleanup
        image: photo-cleanup:latest
        ports:
        - containerPort: 8080
        env:
        - name: ROOT_DIR
          value: "/photos"
        - name: PORT
          value: "8080"
        volumeMounts:
        - name: photos
          mountPath: /photos
          readOnly: true
        - name: data
          mountPath: /app/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: photos
        hostPath:
          path: /mnt/photos  # Adjust to your setup
          type: Directory
      - name: data
        persistentVolumeClaim:
          claimName: photo-cleanup-data
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: photo-cleanup-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: photo-cleanup
spec:
  selector:
    app: photo-cleanup
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
  type: ClusterIP
```

### Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: photo-cleanup
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - photos.example.com
    secretName: photo-cleanup-tls
  rules:
  - host: photos.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: photo-cleanup
            port:
              number: 8080
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_DIR` | `/photos` | Root directory to scan for media files |
| `PORT` | `8080` | Web server port |
| `DATA_DIR` | `/app/data` | Directory for persistent data (thumbnails, embeddings, etc.) |
| `IMMICH_API_URL` | - | Immich API endpoint (optional) |
| `IMMICH_API_KEY` | - | Immich API key (optional) |
| `RUNNING_IN_DOCKER` | - | Set to `true` to force Docker mode |

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs photo-cleanup

# Common issues:
# 1. Invalid ROOT_DIR - ensure the path exists and is mounted
# 2. Port conflict - change PORT in .env file
# 3. Insufficient memory - increase limits in docker-compose.yml
```

### Permission Errors
```bash
# Fix ownership of data directory
docker exec photo-cleanup chown -R appuser:appuser /app/data

# Or mount with correct permissions
# In docker-compose.yml, add user mapping:
services:
  photo-cleanup:
    user: "1000:1000"  # Match your host user ID
```

### Slow Performance
- Increase CPU/memory limits
- Use SSD for data volume
- Enable GPU acceleration (Linux only)
- Reduce concurrent scans

### Health Check Failing
```bash
# Check if app is responding
docker exec photo-cleanup curl http://localhost:8080

# Increase health check timeouts in Dockerfile:
# HEALTHCHECK --start-period=120s ...
```

## Advanced: Custom Base Image

For even smaller images, consider using Alpine:

```dockerfile
FROM python:3.11-alpine

RUN apk add --no-cache \
    ffmpeg \
    build-base \
    jpeg-dev \
    zlib-dev

# ... rest of Dockerfile
```

**Trade-offs:**
- ✅ Smaller image (~600MB vs ~1.2GB)
- ❌ Longer build time
- ❌ Potential compatibility issues
