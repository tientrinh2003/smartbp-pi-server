#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

SmartBP Pi5 Enhanced Server (clean)

- FastAPI server for audio inference with YAMNet (TFLite)

- Optional background microphone capture via sounddevice

- Optional camera snapshot via OpenCV

- Auto-handle Flex delegate if model contains SELECT_TF_OPS

"""



import os
import asyncio

import io

import glob

import sys

import wave

import time

import base64

import logging

import threading

from pathlib import Path

from typing import List, Optional, Tuple, Dict



import numpy as np

from fastapi import FastAPI, UploadFile, File, Response, Body
from pydantic import BaseModel

from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel



# ---- Config ----

MODEL_DIR = Path("/home/tien/smartbp/models")

MODEL_TFLITE = MODEL_DIR / "yamnet_finetuned_builtin.tflite"   # Æ°u tiÃªn builtin-only

MODEL_TFLITE_FALLBACK = MODEL_DIR / "yamnet_finetuned.tflite"  # cÃ³ SELECT_TF_OPS (Flex)

AUDIO_RATE = 16000                      # YAMNet expects 16 kHz mono float32

AUDIO_BG_SECONDS = 2.0                  # window 2.0s (model expects 32000 samples)

AUDIO_BG_ENABLE = True                  # báº­t/táº¯t background audio

CAMERA_INDEX = 0                        # /dev/video0



# ---- Logging ----

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",

    datefmt="%H:%M:%S",

)

logger = logging.getLogger("smartbp")



# ---- Optional imports ----

try:

    import sounddevice as sd

    HAVE_SD = True

except Exception as e:

    logger.warning("sounddevice not available: %s", e)

    HAVE_SD = False



try:

    import cv2

    HAVE_CV2 = True

except Exception as e:

    logger.warning("OpenCV not available: %s", e)

    HAVE_CV2 = False





# ============================================================================

# TFLite Interpreter loader (tflite-runtime first, then TF + Flex delegate)

# ============================================================================



class InferenceBackend:

    def __init__(self):

        self.impl = None               # callable: run_inference(waveform) -> np.ndarray

        self.input_detail = None

        self.output_detail = None

        self.backend_name = "uninitialized"



    def _try_load_tflite_runtime(self, model_path: Path) -> bool:

        """Try to load with tflite-runtime.Interpreter (no Flex)."""

        try:

            from tflite_runtime.interpreter import Interpreter

            logger.info("Loading TFLite (tflite-runtime) from: %s", model_path)

            interp = Interpreter(model_path=str(model_path))

            interp.allocate_tensors()

            in_det = interp.get_input_details()[0]

            out_det = interp.get_output_details()[0]



            def run_inference(waveform: np.ndarray) -> np.ndarray:

                interp.set_tensor(in_det["index"], waveform.astype(np.float32))

                interp.invoke()

                return interp.get_tensor(out_det["index"])



            self.impl = run_inference

            self.input_detail = in_det

            self.output_detail = out_det

            self.backend_name = f"tflite-runtime:{model_path.name}"

            return True

        except Exception as e:

            logger.info("tflite-runtime load failed: %s", e)

            return False



    def _find_flex_delegate(self) -> Optional[str]:

        """Locate TensorFlow Flex delegate shared object."""

        try:

            import site

            site_pkgs = site.getsitepackages()[0]

            cands = glob.glob(os.path.join(site_pkgs, "tensorflow", "lite", "experimental", "*flex*.so"))

            return cands[0] if cands else None

        except Exception:

            return None



    def _try_load_tf_with_flex(self, model_path: Path) -> bool:

        """Try to load with TF's TFLite Interpreter + Flex delegate."""

        try:

            import tensorflow as tf

            from tensorflow.lite.python.interpreter import Interpreter, load_delegate



            flex_lib = self._find_flex_delegate()

            delegates = []

            if flex_lib:

                logger.info("Using Flex delegate: %s", flex_lib)

                delegates = [load_delegate(flex_lib)]

            else:

                logger.warning("Flex delegate not found â€” will try without it (may fail).")



            logger.info("Loading TFLite (TF interpreter) from: %s", model_path)

            interp = Interpreter(model_path=str(model_path), experimental_delegates=delegates)

            interp.allocate_tensors()

            in_det = interp.get_input_details()[0]

            out_det = interp.get_output_details()[0]



            def run_inference(waveform: np.ndarray) -> np.ndarray:

                interp.set_tensor(in_det["index"], waveform.astype(np.float32))

                interp.invoke()

                return interp.get_tensor(out_det["index"])



            self.impl = run_inference

            self.input_detail = in_det

            self.output_detail = out_det

            self.backend_name = f"tf-lite-flex:{model_path.name}"

            return True

        except Exception as e:

            logger.error("TF+Flex interpreter load failed: %s", e)

            return False



    def load(self) -> None:

        """Load model in priority order."""

        # 1) builtin-only preferred

        if MODEL_TFLITE.exists() and self._try_load_tflite_runtime(MODEL_TFLITE):

            return

        # 2) fallback model (may need Flex)

        if MODEL_TFLITE_FALLBACK.exists() and self._try_load_tflite_runtime(MODEL_TFLITE_FALLBACK):

            return

        # 3) try TF interpreter + Flex for each

        if MODEL_TFLITE.exists() and self._try_load_tf_with_flex(MODEL_TFLITE):

            return

        if MODEL_TFLITE_FALLBACK.exists() and self._try_load_tf_with_flex(MODEL_TFLITE_FALLBACK):

            return

        raise RuntimeError("No TFLite model could be loaded with available backends.")



    def run(self, waveform: np.ndarray) -> np.ndarray:

        if self.impl is None:

            raise RuntimeError("Inference backend not initialized")

        return self.impl(waveform)





backend = InferenceBackend()





# ============================================================================

# Audio utils: WAV decode, resample to 16k, normalization

# ============================================================================



def read_wav_pcm16_to_float32(data: bytes) -> Tuple[np.ndarray, int]:

    """

    Parse a WAV PCM16 mono/stereo into float32 [-1, 1], return (waveform, sample_rate).

    """

    with wave.open(io.BytesIO(data), "rb") as wf:

        n_channels = wf.getnchannels()

        sampwidth = wf.getsampwidth()

        framerate = wf.getframerate()

        n_frames = wf.getnframes()



        if sampwidth != 2:

            raise ValueError("Only PCM16 WAV is supported (sampwidth=2)")

        frames = wf.readframes(n_frames)

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if n_channels == 2:

            audio = audio.reshape(-1, 2).mean(axis=1)

        return audio, framerate





def simple_resample_1d(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:

    """Linear resample (no dependency on librosa/scipy)."""

    if src_rate == dst_rate:

        return x

    duration = len(x) / float(src_rate)

    t_src = np.linspace(0.0, duration, num=len(x), endpoint=False)

    n_dst = int(round(duration * dst_rate))

    t_dst = np.linspace(0.0, duration, num=n_dst, endpoint=False)

    return np.interp(t_dst, t_src, x).astype(np.float32)





def pad_or_trim_to_seconds(x: np.ndarray, seconds: float, rate: int) -> np.ndarray:

    target = int(round(seconds * rate))

    if len(x) == target:

        return x

    if len(x) > target:

        return x[:target]

    # pad at end

    out = np.zeros((target,), dtype=np.float32)

    out[:len(x)] = x

    return out





def topk_indices(scores: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:

    """Return top-k (index, prob) pairs from scores (1D)."""

    scores = scores.flatten()

    idx = np.argsort(scores)[::-1][:k]

    return [(int(i), float(scores[i])) for i in idx]





# ============================================================================

# Background audio thread (optional)

# ============================================================================



class AudioState:

    def __init__(self):

        self.lock = threading.Lock()

        self.buffer = np.array([], dtype=np.float32)

        self.last_scores: List[Tuple[int, float]] = []

        self.last_inference_at: float = 0.0  # timestamp of last successful inference

        self.running = False

        self.thread: Optional[threading.Thread] = None





audio_state = AudioState()

# ---- Camera State Management ----
class CameraState:
    def __init__(self):
        self.streaming_active = False
        self.lock = threading.Lock()
        self.active_captures = []  # Track active VideoCapture objects
    
    def set_streaming(self, active: bool):
        with self.lock:
            if active:
                self.streaming_active = True
            else:
                # Ensure all captures are released before marking as inactive
                for cap in self.active_captures:
                    try:
                        if cap and cap.isOpened():
                            cap.release()
                            logger.info("📷 Force released camera capture")
                    except Exception as e:
                        logger.error(f"Error releasing camera: {e}")
                self.active_captures.clear()
                self.streaming_active = False
                logger.info("📷 Camera streaming stopped")
    
    def register_capture(self, cap):
        with self.lock:
            self.active_captures.append(cap)
    
    def unregister_capture(self, cap):
        with self.lock:
            if cap in self.active_captures:
                self.active_captures.remove(cap)

camera_state = CameraState()

def sd_callback(indata, frames, time_info, status):

    if status:

        logger.warning("sounddevice status: %s", status)
    
    # Audio callback - logging disabled to reduce spam

    with audio_state.lock:

        audio_state.buffer = np.append(audio_state.buffer, indata[:, 0].astype(np.float32))





def audio_loop():

    logger.info("ðŸŽ™ï¸  Background audio started (%.1fs window @ %d Hz).", AUDIO_BG_SECONDS, AUDIO_RATE)

    buf_needed = int(AUDIO_RATE * AUDIO_BG_SECONDS)

    stream = None

    try:

        # Auto-detect the first available microphone input device
        devices = sd.query_devices()
        input_device = None
        
        # Find first device with input channels (microphone)
        for i, device_info in enumerate(devices):
            if device_info['max_input_channels'] > 0:
                input_device = i
                logger.info(f"🎤 Using microphone: {device_info['name']} (device {i})")
                break
        
        if input_device is None:
            logger.warning("No input audio device found, using default")
            input_device = None  # Let sounddevice choose default
            
        stream = sd.InputStream(

            samplerate=AUDIO_RATE,

            channels=1,

            dtype="float32",

            callback=sd_callback,
            
            device=input_device,  # Auto-detected or default

        )

        stream.start()

        # Status reporting counter
        status_counter = 0
        
        while audio_state.running:

            wf = None
            status_counter += 1

            with audio_state.lock:
                current_buffer_size = len(audio_state.buffer)
                if current_buffer_size >= buf_needed:

                    wf = audio_state.buffer[:buf_needed].copy()

                    audio_state.buffer = audio_state.buffer[buf_needed:]
                    # Processing inference - logging disabled to reduce spam
                else:
                    # Show buffer status every 20 cycles (~1 second)
                    if status_counter % 20 == 0:
                        logger.info(f"📊 Buffer: {current_buffer_size}/{buf_needed} ({current_buffer_size/buf_needed*100:.0f}%) - waiting...")

            if wf is not None:

                # Infer with real audio data

                scores = backend.run(np.expand_dims(wf, axis=0))
                
                # Process results and update state
                results = topk_indices(scores, k=5)
                top_class, top_score = results[0]
                
                # Debug: show raw scores and shape
                raw_scores = scores[0]  # Get first batch
                logger.info(f"🔍 Model output shape: {raw_scores.shape}, values: {raw_scores}")
                
                # Handle different output shapes
                if len(raw_scores) == 1:
                    # Binary classification with single output (sigmoid)
                    speech_prob = float(raw_scores[0])
                    speech_status = "SPEECH" if speech_prob > 0.5 else "NO_SPEECH"
                    confidence = f"{speech_prob:.3f}"
                elif len(raw_scores) >= 2:
                    # Multi-class with softmax
                    logger.info(f"🔍 Raw scores - class 0 (no_speech): {raw_scores[0]:.6f}, class 1 (speech): {raw_scores[1]:.6f}")
                    speech_status = "SPEECH" if top_class == 1 else "NO_SPEECH"
                    confidence = f"{top_score:.3f}"
                else:
                    speech_status = "UNKNOWN"
                    confidence = "0.000"
                
                logger.info(f"🎤 Speech Detection: {speech_status} (confidence: {confidence})")

                with audio_state.lock:
                    if len(raw_scores) == 1:
                        # Single output (sigmoid) - convert to binary classification format
                        speech_prob = float(raw_scores[0])
                        if speech_prob > 0.5:
                            # Speech detected - class 1 with high confidence
                            audio_state.last_scores = [(1, speech_prob), (0, 1.0-speech_prob)]
                            logger.info(f"🔄 Converted to: class 1 (speech) = {speech_prob:.3f}, class 0 = {1.0-speech_prob:.3f}")
                        else:
                            # No speech - class 0 with high confidence
                            audio_state.last_scores = [(0, 1.0-speech_prob), (1, speech_prob)]
                            logger.info(f"🔄 Converted to: class 0 (no_speech) = {1.0-speech_prob:.3f}, class 1 = {speech_prob:.3f}")
                    else:
                        audio_state.last_scores = results
                    audio_state.last_inference_at = time.time()

            else:

                # Not enough buffered audio yet; keep previous inference scores.
                # Initialize baseline only if we have never inferred.
                with audio_state.lock:
                    if not audio_state.last_scores:
                        audio_state.last_scores = [(0, 0.99), (1, 0.01)]

            time.sleep(0.05)

    except Exception as e:

        logger.error("Audio loop error: %s", e)

    finally:

        if stream:

            stream.stop()

            stream.close()

        logger.info("ðŸŽ™ï¸  Background audio stopped.")





def start_audio_if_possible():

    if not AUDIO_BG_ENABLE:

        logger.info("Background audio disabled by config.")

        return

    if not HAVE_SD:

        logger.warning("sounddevice not available; skip background audio.")

        return

    if audio_state.running:

        logger.info("Audio already running.")

        return

    # Add small delay to ensure hardware is ready after previous stop
    time.sleep(0.1)

    audio_state.running = True

    audio_state.thread = threading.Thread(target=audio_loop, daemon=True)

    audio_state.thread.start()





def stop_audio():

    if not audio_state.running:

        logger.info("Audio already stopped.")

        return

    audio_state.running = False

    if audio_state.thread:

        audio_state.thread.join(timeout=2.0)

        audio_state.thread = None

    # Add delay to ensure hardware cleanup before next start
    time.sleep(0.15)





# ============================================================================

# Camera snapshot

# ============================================================================



def capture_jpeg() -> Optional[bytes]:

    if not HAVE_CV2:

        return None

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():

        return None

    try:

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ok, frame = cap.read()

        if not ok:

            return None

        ok, buf = cv2.imencode(".jpg", frame)

        if not ok:

            return None

        return buf.tobytes()

    finally:

        cap.release()





# ============================================================================

# FastAPI app

# ============================================================================



app = FastAPI(title="SmartBP Pi5 Enhanced", version="1.0.0")

# Add CORS middleware to allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.22.65:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferResponse(BaseModel):

    backend: str

    topk: List[Tuple[int, float]]





@app.on_event("startup")

def on_startup():

    # Load interpreter

    backend.load()

    logger.info("Inference backend ready: %s", backend.backend_name)

    # Audio will be started only when monitoring is requested
    logger.info("🎯 Audio monitoring ready - waiting for start command")





@app.on_event("shutdown")

def on_shutdown():

    stop_audio()

    logger.info("Server shutdown")





@app.get("/", response_class=HTMLResponse)

def root_page():

    html = f"""

    <html>

      <head><title>SmartBP Pi5 Enhanced</title></head>

      <body style="font-family: sans-serif;">

        <h2>SmartBP Pi5 Enhanced</h2>

        <ul>

          <li>Backend: <b>{backend.backend_name}</b></li>

          <li><a href="/api/health">/api/health</a></li>

          <li><a href="/api/model">/api/model</a></li>

          <li><a href="/api/speech/status">/api/speech/status</a></li>

          <li>POST /api/speech/infer (multipart/form-data, file field: <code>file</code>, WAV PCM16)</li>

          <li><a href="/api/camera/snapshot">/api/camera/snapshot</a> (if OpenCV & camera available)</li>
          <li><a href="/api/camera/stream">/api/camera/stream</a> (MJPEG stream with AI overlay)</li>

        </ul>

        <p>Tip: Use <code>curl</code> to test:</p>

        <pre>

curl -F "file=@sample.wav" http://localhost:8000/api/speech/infer

        </pre>

      </body>

    </html>

    """

    return HTMLResponse(html)





@app.get("/api/health")

def api_health():

    return {"ok": True, "backend": backend.backend_name}





@app.get("/api/model")

def api_model():

    return {

        "backend": backend.backend_name,

        "input": backend.input_detail,

        "output": backend.output_detail,

        "expects_rate": AUDIO_RATE,

        "window_seconds": AUDIO_BG_SECONDS,

    }





@app.get("/api/speech/status", response_model=InferResponse)

def api_speech_status():

    with audio_state.lock:

        topk = list(audio_state.last_scores)

    return InferResponse(backend=backend.backend_name, topk=topk)


# -----------------------
# Bluetooth measurement background manager
# -----------------------
# Mapping: device_address -> {"task": asyncio.Task, "cancel": asyncio.Event}
bluetooth_tasks: Dict[str, Dict] = {}


async def _run_measure_background(device_address: str, timeout: int = 120, cancel_event: asyncio.Event = None):
    try:
        # import here to avoid circular / startup issues
        from bluetooth_bp_client import measure_once
        # propagate_exceptions=True so we can react to real errors
        result = await measure_once(device_address, timeout=timeout, cancel_event=cancel_event, propagate_exceptions=True)
        logger.info(f"Background measure finished for {device_address}: {result}")
        return result
    except asyncio.CancelledError:
        logger.info(f"Background measure task cancelled for {device_address}")
        raise
    except Exception as e:
        logger.error(f"Background measure error for {device_address}: {e}")
        # On measurement error, cancel other bluetooth measurement tasks only
        try:
            cancel_all_bluetooth_tasks()
        except Exception as stop_err:
            logger.error(f"Error while cancelling bluetooth tasks: {stop_err}")
    finally:
        # cleanup
        bluetooth_tasks.pop(device_address, None)


def stop_all_bluetooth_measurements():
    """Signal cancellation and cancel all running bluetooth measurement tasks."""
    logger.info("🔴 Stopping all bluetooth measurements due to error or explicit stop request")
    # Make a shallow copy to avoid modification during iteration
    for addr, entry in list(bluetooth_tasks.items()):
        try:
            entry.get("cancel") and entry["cancel"].set()
            task = entry.get("task")
            if task and not task.done():
                task.cancel()
                logger.info(f"  - Cancelled task for {addr}")
        except Exception as e:
            logger.error(f"Failed to cancel task for {addr}: {e}")
    # NOTE: do not stop audio or shutdown server here; keep audio/camera running


def cancel_all_bluetooth_tasks():
    """Cancel bluetooth measurement tasks but do not touch audio/camera/system state."""
    logger.info("🟠 Cancelling all bluetooth measurement tasks (preserve audio/camera)")
    for addr, entry in list(bluetooth_tasks.items()):
        try:
            entry.get("cancel") and entry["cancel"].set()
            task = entry.get("task")
            if task and not task.done():
                task.cancel()
                logger.info(f"  - Cancelled task for {addr}")
        except Exception as e:
            logger.error(f"Failed to cancel task for {addr}: {e}")


@app.post("/api/bluetooth/measure/start")
async def api_bluetooth_measure_start(body: Dict = Body(...)):
    device_address = body.get("device_address")
    timeout = int(body.get("timeout", 120))
    if not device_address:
        return JSONResponse({"error": "missing device_address"}, status_code=400)

    if device_address in bluetooth_tasks:
        return JSONResponse({"status": "already_running"}, status_code=409)

    cancel_event = asyncio.Event()
    task = asyncio.create_task(_run_measure_background(device_address, timeout=timeout, cancel_event=cancel_event))
    bluetooth_tasks[device_address] = {"task": task, "cancel": cancel_event}
    logger.info(f"Started background measurement for {device_address}")
    return {"status": "started"}


@app.post("/api/bluetooth/measure/stop")
async def api_bluetooth_measure_stop(body: Dict = Body(...)):
    device_address = body.get("device_address")
    if not device_address:
        return JSONResponse({"error": "missing device_address"}, status_code=400)
    # Special token to stop all
    if device_address == 'all':
        cancel_all_bluetooth_tasks()
        logger.info("Stop requested for all bluetooth measurements")
        return {"status": "stopping_all"}

    entry = bluetooth_tasks.get(device_address)
    if not entry:
        return JSONResponse({"status": "not_running"}, status_code=404)

    # Signal cancellation and cancel task
    try:
        entry["cancel"].set()
        entry["task"].cancel()
        logger.info(f"Stop requested for background measurement {device_address}")
        return {"status": "stopping"}
    except Exception as e:
        logger.error(f"Error stopping measurement for {device_address}: {e}")
        return JSONResponse({"error": "failed_to_stop"}, status_code=500)



@app.get("/api/ai/status")
def api_ai_status():
    """Get current AI speech detection status"""
    
    with audio_state.lock:
        topk = list(audio_state.last_scores)
    
    # Check if we have valid data
    if not topk or len(topk) == 0:
        logger.info("🎤 API Status: no_data")
        logger.info("📡 API Response: is_speaking=False")
        return JSONResponse({
            "is_speaking": False,
            "confidence": 0.0,
            "status": "no_data",
            "timestamp": time.time()
        })
    
    # Binary model: topk = [(class_idx, probability)]
    top_class, top_prob = topk[0]
    
    # class 0 = no_speech, class 1 = speech
    is_speaking = (top_class == 1)
    confidence = top_prob
    
    status = "speaking" if is_speaking else "quiet"
    
    logger.info(f"🎤 API Status: {status} (confidence: {confidence:.3f}, class: {top_class})")
    logger.info(f"📡 API Response: is_speaking={is_speaking}")
    
    return JSONResponse({
        "is_speaking": is_speaking,
        "confidence": confidence,
        "status": status,
        "timestamp": time.time()
    })





@app.post("/api/speech/infer", response_model=InferResponse)

async def api_speech_infer(file: UploadFile = File(...)):

    data = await file.read()

    wav, rate = read_wav_pcm16_to_float32(data)

    wav = simple_resample_1d(wav, rate, AUDIO_RATE)

    wav = pad_or_trim_to_seconds(wav, seconds=2.0, rate=AUDIO_RATE)  # 2s for 32000 samples

    scores = backend.run(np.expand_dims(wav, axis=0))

    topk = topk_indices(scores, k=5)

    return InferResponse(backend=backend.backend_name, topk=topk)





@app.get("/api/camera/snapshot")

def api_camera_snapshot():

    img = capture_jpeg()

    if img is None:

        return JSONResponse({"error": "Camera not available"}, status_code=404)

    return Response(content=img, media_type="image/jpeg")


def get_current_ai_status():
    """Helper function to get current AI status for camera overlay"""
    try:
        with audio_state.lock:
            topk = list(audio_state.last_scores)
        
        if not topk or len(topk) == 0:
            return {
                "speech": {
                    "status": "no_data",
                    "confidence": 0.0,
                    "timestamp": time.time(),
                    "color": "green",
                    "is_speaking": False
                }
            }
        
        # Binary model: topk = [(class_idx, probability)]
        top_class, top_prob = topk[0]
        
        # class 0 = no_speech, class 1 = speech
        is_speaking = (top_class == 1)
        confidence = top_prob
        
        if is_speaking:
            if confidence > 0.8:
                status = "speaking"
                color = "red"
            else:
                status = "maybe_speaking"
                color = "yellow"
        else:
            status = "quiet"
            color = "green"
        
        return {
            "speech": {
                "status": status,
                "confidence": confidence,
                "timestamp": time.time(),
                "color": color,
                "is_speaking": is_speaking
            }
        }
    except Exception as e:
        logger.error(f"Error getting AI status: {e}")
        return None


@app.post("/api/monitoring/start")
def start_monitoring():
    """Start audio and camera monitoring"""
    global audio_state
    try:
        if not audio_state.running:
            # Start audio monitoring using existing function
            start_audio_if_possible()
            logger.info("📢 Monitoring started - audio and camera active")
        else:
            logger.info("📢 Monitoring already running")
        
        return {"success": True, "message": "Monitoring started", "services": ["audio", "camera"]}
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/monitoring/stop") 
def stop_monitoring():
    """Stop audio monitoring and camera streaming"""
    global audio_state, camera_state
    try:
        # Stop audio using existing function
        stop_audio()
        # Signal camera to stop streaming
        camera_state.set_streaming(False)
        logger.info("🔇 Monitoring stopped - audio and camera disabled")
        return {"success": True, "message": "Monitoring stopped", "services": ["audio", "camera"]}
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/bluetooth/scan")
async def scan_bluetooth():
    """Scan for nearby Bluetooth devices"""
    try:
        from bluetooth_scanner import scan_bluetooth_devices
        devices = await scan_bluetooth_devices(scan_duration=8)
        return {"success": True, "devices": devices, "count": len(devices)}
    except Exception as e:
        logger.error(f"Bluetooth scan error: {e}")
        return {"success": False, "error": str(e), "devices": []}


class MeasureRequest(BaseModel):
    device_address: str

@app.post("/api/bluetooth/measure")
async def measure_blood_pressure(request: MeasureRequest):
    """
    Measure blood pressure from Bluetooth device
    Body: { "device_address": "00:5F:BF:3A:51:BD" }
    """
    try:
        device_address = request.device_address
        
        if not device_address:
            return JSONResponse({
                "success": False, 
                "error": "Missing device_address"
            }, status_code=400)
        
        logger.info(f"🩺 Bắt đầu đo huyết áp từ thiết bị: {device_address}")
        
        from bluetooth_bp_client import measure_once
        result = await measure_once(device_address, timeout=120)
        
        if result:
            logger.info(f"✅ API trả về kết quả: {result}")
            return JSONResponse({
                "success": True,
                "data": result,
                "message": "Đo huyết áp thành công"
            })
        else:
            logger.warning("⚠️ Không nhận được dữ liệu từ thiết bị")
            return JSONResponse({
                "success": False, 
                "error": "Không nhận được dữ liệu. Vui lòng đảm bảo đã BẤM NÚT START trên máy Omron sau khi kết nối."
            }, status_code=408)
    
    except Exception as e:
        logger.error(f"Measurement error: {e}")
        return JSONResponse({
            "success": False, 
            "error": str(e)
        }, status_code=500)


@app.get("/api/camera/stream")
def api_camera_stream():
    """MJPEG streaming endpoint for camera with AI overlay"""
    if not HAVE_CV2:
        return JSONResponse({"error": "OpenCV not available for streaming"}, status_code=404)
    
    # Check if camera is already streaming
    with camera_state.lock:
        if camera_state.streaming_active:
            logger.warning("⚠️ Camera already streaming - rejecting duplicate request")
            return JSONResponse({"error": "Camera already in use by another client"}, status_code=409)
        # Set streaming_active BEFORE opening camera to prevent race condition
        camera_state.streaming_active = True
    
    def generate_mjpeg():
        import time
        logger.info("📹 Camera streaming started with AI overlay")
        
        cap = None
        try:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                logger.error("Cannot open camera for streaming")
                camera_state.set_streaming(False)  # Reset flag on failure
                return
            
            # Register this capture object for cleanup
            camera_state.register_capture(cap)
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 20)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
            
            frame_count = 0
            # Note: streaming_active already set to True before opening camera
            last_ai_check = 0
            cached_ai_data = None
            
            while camera_state.streaming_active:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Get AI status only every 0.5s to reduce overhead
                current_time = time.time()
                if current_time - last_ai_check > 0.5:
                    try:
                        cached_ai_data = get_current_ai_status()
                        last_ai_check = current_time
                    except Exception as e:
                        logger.error(f"Error getting AI status: {e}")
                        cached_ai_data = None
                
                # Add AI overlay with cached data
                try:
                    if cached_ai_data and cached_ai_data.get('speech'):
                        speech = cached_ai_data['speech']
                        status = speech.get('status', 'unknown')
                        confidence = speech.get('confidence', 0.0)
                        is_speaking = speech.get('is_speaking', False)
                        
                        # Simpler overlay for better performance
                        if is_speaking:
                            text = f"Dang noi ({confidence:.0%})"
                            color = (0, 0, 255)  # Red
                        else:
                            text = f"Im lang ({confidence:.0%})"
                            color = (0, 255, 0)  # Green
                        
                        # Draw text with shadow for visibility
                        cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, (0, 0, 0), 3)  # Shadow
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, color, 2)  # Text
                        
                except Exception as e:
                    # Silently fail overlay to not interrupt stream
                    pass
                
                # Encode frame as JPEG with lower quality for speed
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ret:
                    break
                    
                frame_bytes = buffer.tobytes()
                
                # Yield MJPEG frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_count += 1
                time.sleep(0.03)  # ~30 FPS max (reduced sleep for smoother video)
                
        except GeneratorExit:
            # Client disconnected - clean up immediately
            logger.info("📷 Client disconnected from stream")
        except Exception as e:
            logger.error(f"Camera streaming error: {e}")
        finally:
            if cap is not None:
                camera_state.unregister_capture(cap)
                if cap.isOpened():
                    cap.release()
                    logger.info("📷 Camera released")
            camera_state.set_streaming(False)
            logger.info("📹 Camera streaming ended")
    
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================================

# Entrypoint (uvicorn)

# ============================================================================

def check_system_status():
    """Check and report system component status."""
    logger.info("🔍 System Component Check:")
    
    # Check model file
    model_found = False
    if MODEL_TFLITE.exists():
        logger.info(f"✅ Model file ready: {MODEL_TFLITE} ({MODEL_TFLITE.stat().st_size / 1024 / 1024:.1f}MB)")
        model_found = True
    elif MODEL_TFLITE_FALLBACK.exists():
        logger.info(f"✅ Fallback model ready: {MODEL_TFLITE_FALLBACK} ({MODEL_TFLITE_FALLBACK.stat().st_size / 1024 / 1024:.1f}MB)")
        model_found = True
    else:
        logger.error(f"❌ No model files found: {MODEL_TFLITE} or {MODEL_TFLITE_FALLBACK}")
    
    # Check camera
    if HAVE_CV2:
        try:
            test_cap = cv2.VideoCapture(CAMERA_INDEX)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    logger.info(f"✅ Camera ready: /dev/video{CAMERA_INDEX} ({frame.shape[1]}x{frame.shape[0]})")
                else:
                    logger.error("❌ Camera capture test failed")
                test_cap.release()
            else:
                logger.error(f"❌ Camera not accessible: /dev/video{CAMERA_INDEX}")
        except Exception as e:
            logger.error(f"❌ Camera error: {e}")
    else:
        logger.error("❌ OpenCV not available")
    
    # Check microphone
    if HAVE_SD:
        try:
            devices = sd.query_devices()
            mic_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            if mic_devices:
                logger.info(f"✅ Microphone ready: {len(mic_devices)} input device(s) available")
                for i in mic_devices[:2]:  # Show first 2 devices
                    logger.info(f"   📱 Device {i}: {devices[i]['name']} ({int(devices[i]['default_samplerate'])}Hz)")
            else:
                logger.error("❌ No microphone devices found")
        except Exception as e:
            logger.error(f"❌ Microphone error: {e}")
    else:
        logger.error("❌ sounddevice not available")


if __name__ == "__main__":

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")

    port = int(os.environ.get("PORT", "8000"))

    reload_flag = False  # Ä‘á»ƒ False trÃªn Pi cho nháº¹

    logger.info("ðŸš€ Starting server on %s:%d", host, port)

    uvicorn.run("smartbp_pi5_enhanced:app", host=host, port=port, reload=reload_flag)