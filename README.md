# SmartBP Raspberry Pi Server

Enhanced FastAPI server for SmartBP system with real-time speech detection and camera streaming.

## Features

- **Real-time Speech Detection**: YamNet fine-tuned binary model for speech/no-speech classification
- **Camera Streaming**: MJPEG streaming with AI overlay showing speech detection status
- **Monitoring Control**: Start/stop APIs for complete monitoring lifecycle management
- **Hardware Integration**: Auto-detection of microphone and camera devices
- **CORS Support**: Proper CORS handling for web integration

## Hardware Requirements

- Raspberry Pi 5 (recommended) or Pi 4
- Logitech Webcam C925e (or compatible USB camera)
- Microphone (built-in webcam mic or external USB mic)

## Software Dependencies

- Python 3.11+
- FastAPI + Uvicorn
- OpenCV (cv2)
- sounddevice
- TensorFlow Lite Runtime
- NumPy

## Installation

1. Set up virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install fastapi uvicorn opencv-python sounddevice numpy
pip install tflite-runtime  # For Pi optimization
```

3. Setup YamNet model files:
```bash
# Run model setup script
chmod +x download_models.sh
./download_models.sh

# Or manually copy your trained models to models/ directory
# See models/README.md for detailed instructions
```

## Usage

1. Start server:
```bash
python smartbp_pi5_enhanced.py
```

2. Server will be available at `http://0.0.0.0:8000`

## API Endpoints

- `GET /api/health` - Server health check
- `GET /api/ai/status` - Current speech detection status
- `POST /api/monitoring/start` - Start audio monitoring
- `POST /api/monitoring/stop` - Stop monitoring and camera
- `GET /api/camera/stream` - MJPEG camera stream with AI overlay
- `GET /api/camera/test` - Test camera availability

## Configuration

Key settings in the code:
- `AUDIO_RATE = 16000` - Audio sampling rate for YamNet
- `AUDIO_BG_SECONDS = 2.0` - Analysis window size
- `CAMERA_INDEX = 0` - Camera device index (/dev/video0)

## Model Information

The system uses a fine-tuned YamNet binary classification model:
- **Input**: 32,000 samples (2 seconds at 16kHz)
- **Output**: Single sigmoid value (speech probability)
- **Threshold**: 0.5 for speech/no-speech classification

## Development

For development with auto-reload:
```bash
uvicorn smartbp_pi5_enhanced:app --host 0.0.0.0 --port 8000 --reload
```

## License

MIT License - See LICENSE file for details