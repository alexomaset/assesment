# REM Waste Accent Analyzer - Production Deployment Guide

This guide outlines how to deploy the REM Waste Accent Analyzer application to various production environments.

## Prerequisites

- Docker and Docker Compose (for containerized deployment)
- Git
- Python 3.8+ (for local testing)
- Access to a cloud platform (optional)

## Deployment Options

### 1. Docker Deployment (Recommended)

The application is containerized and ready for deployment using Docker:

```bash
# Clone the repository (if deploying from source)
git clone <your-repository-url>
cd remwaste-accent-analyzer

# Build and start the application
docker-compose up -d

# Check logs
docker-compose logs -f
```

The application will be available at `http://your-server-ip:8501`

#### Environment Configuration

Edit the environment variables in `docker-compose.yml` to configure:
- Cache TTL
- Maximum upload size
- Other application settings

### 2. Streamlit Cloud Deployment

For quick deployment without infrastructure management:

1. Push your code to GitHub
2. Sign up at [streamlit.io](https://streamlit.io)
3. Connect to your repository
4. Configure:
   - Python version: 3.9
   - Requirements: `requirements.txt`
   - Main file: `main.py`

### 3. Cloud Platform Deployment

#### AWS Elastic Beanstalk

1. Install EB CLI: `pip install awsebcli`
2. Initialize EB: `eb init`
3. Create environment: `eb create`
4. Deploy: `eb deploy`

#### Google Cloud Run

1. Install Google Cloud SDK
2. Build and deploy:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/accent-analyzer
gcloud run deploy accent-analyzer --image gcr.io/YOUR_PROJECT_ID/accent-analyzer --platform managed
```

## Optimizations

The application includes several production-ready optimizations:

1. **Model Quantization**: Reduces model size by ~75% and improves inference speed
2. **Input Length Limiting**: 5-second max audio processing for consistent performance
3. **Async Processing**: Parallel audio chunk processing with UI progress updates
4. **Multi-level Caching**: URL, video, and chunk-based caching with 24-hour expiration

## Monitoring and Maintenance

### Logs

When using Docker, view logs with:
```bash
docker-compose logs -f
```

### Updates

To update the application:
```bash
git pull
docker-compose down
docker-compose up -d --build
```

### Backup

Regularly backup the cache directory to preserve processed data:
```bash
# Example backup script
tar -czf accent-analyzer-cache-$(date +%Y%m%d).tar.gz ./cache
```

## Security Considerations

1. Run the application behind a reverse proxy (Nginx, Apache)
2. Enable HTTPS
3. Consider implementing authentication for the Streamlit interface
4. Keep dependencies updated with:
   ```bash
   pip install -U -r requirements.txt
   ```

## Scaling

For high-traffic scenarios:
1. Deploy behind a load balancer
2. Use a distributed cache (Redis) instead of file-based caching
3. Consider separating the model inference into a separate microservice
