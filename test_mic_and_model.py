#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Mic + YAMNet Model (Raspberry Pi Standalone)
================================================
Giống code gốc từ smartbp_pi5_enhanced.py
Chỉ mic + model inference, không có web server
Chạy trên Raspberry Pi và hiển thị kết quả real-time
"""

import os
import sys
import glob
import time
import numpy as np
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smartbp_test")

# ---- Config ----
MODEL_DIR = Path("/home/tien/smartbp/models")
MODEL_TFLITE = MODEL_DIR / "yamnet_finetuned.tflite"  # Chỉ dùng file này
AUDIO_RATE = 16000
AUDIO_BG_SECONDS = 2.0
AUDIO_BG_ENABLE = True

# ---- Optional imports ----
try:
    import sounddevice as sd
    HAVE_SD = True
except ImportError:
    HAVE_SD = False

# ============================================================================
# Model Loading
# ============================================================================
class InferenceBackend:
    def __init__(self):
        self.impl = None
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
                logger.warning("Flex delegate not found – will try without it.")
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
        if MODEL_TFLITE.exists() and self._try_load_tflite_runtime(MODEL_TFLITE):
            return
        if MODEL_TFLITE.exists() and self._try_load_tf_with_flex(MODEL_TFLITE):
            return
        raise RuntimeError("No TFLite model could be loaded with available backends.")

    def run(self, waveform: np.ndarray) -> np.ndarray:
        if self.impl is None:
            raise RuntimeError("Inference backend not initialized")
        return self.impl(waveform)

backend = InferenceBackend()

# ============================================================================
# Audio Utils
# ============================================================================
def topk_indices(scores: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    """Return top-k (index, prob) pairs from scores (1D)."""
    scores = scores.flatten()
    idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in idx]

# ============================================================================
# Audio State & Threading
# ============================================================================
class AudioState:
    def __init__(self):
        self.lock = threading.Lock()
        self.buffer = np.array([], dtype=np.float32)
        self.last_scores: List[Tuple[int, float]] = []
        self.last_inference_at: float = 0.0
        self.running = False
        self.thread: Optional[threading.Thread] = None

audio_state = AudioState()

def sd_callback(indata, frames, time_info, status):
    if status:
        logger.warning("sounddevice status: %s", status)
    with audio_state.lock:
        audio_state.buffer = np.append(audio_state.buffer, indata[:, 0].astype(np.float32))

def audio_loop():
    logger.info("🎙️ Background audio started (%.1fs window @ %d Hz).", AUDIO_BG_SECONDS, AUDIO_RATE)
    buf_needed = int(AUDIO_RATE * AUDIO_BG_SECONDS)
    stream = None
    try:
        # Auto-detect microphone
        devices = sd.query_devices()
        input_device = None
        for i, device_info in enumerate(devices):
            if device_info['max_input_channels'] > 0:
                input_device = i
                logger.info(f"🎤 Using microphone: {device_info['name']} (device {i})")
                break
        if input_device is None:
            logger.warning("No input audio device found, using default")
            input_device = None

        stream = sd.InputStream(
            samplerate=AUDIO_RATE,
            channels=1,
            dtype="float32",
            callback=sd_callback,
            device=input_device,
        )
        stream.start()
        status_counter = 0
        
        while audio_state.running:
            wf = None
            status_counter += 1
            with audio_state.lock:
                current_buffer_size = len(audio_state.buffer)
                if current_buffer_size >= buf_needed:
                    wf = audio_state.buffer[:buf_needed].copy()
                    audio_state.buffer = audio_state.buffer[buf_needed:]
                else:
                    if status_counter % 20 == 0:
                        logger.info(f"📊 Buffer: {current_buffer_size}/{buf_needed} ({current_buffer_size/buf_needed*100:.0f}%) - waiting...")

            if wf is not None:
                # Timing: Start total measurement
                t_start_total = time.perf_counter()
                
                # Timing: Pre-processing
                t_start_prep = time.perf_counter()
                model_input = np.expand_dims(wf, axis=0)
                t_prep = (time.perf_counter() - t_start_prep) * 1000  # ms
                
                # Timing: Model inference
                t_start_infer = time.perf_counter()
                scores = backend.run(model_input)
                t_infer = (time.perf_counter() - t_start_infer) * 1000  # ms
                
                # Timing: Post-processing
                t_start_post = time.perf_counter()
                results = topk_indices(scores, k=5)
                top_class, top_score = results[0]
                raw_scores = scores[0]
                t_post = (time.perf_counter() - t_start_post) * 1000  # ms
                
                t_total = (time.perf_counter() - t_start_total) * 1000  # ms
                
                logger.info(f"⏱️  TIMING: Total={t_total:.1f}ms | Prep={t_prep:.1f}ms | Inference={t_infer:.1f}ms | Post={t_post:.1f}ms")
                # logger.info(f"🔍 Model output shape: {raw_scores.shape}, values: {raw_scores}")
                
                # Handle different output shapes
                if len(raw_scores) == 1:
                    speech_prob = float(raw_scores[0])
                    speech_status = "SPEECH" if speech_prob > 0.5 else "NO_SPEECH"
                    confidence = f"{speech_prob:.3f}"
                elif len(raw_scores) >= 2:
                    logger.info(f"🔍 Raw scores - class 0: {raw_scores[0]:.6f}, class 1: {raw_scores[1]:.6f}")
                    speech_status = "SPEECH" if top_class == 1 else "NO_SPEECH"
                    confidence = f"{top_score:.3f}"
                else:
                    speech_status = "UNKNOWN"
                    confidence = "0.000"
                
                logger.info(f"🎤 Speech Detection: {speech_status} (confidence: {confidence})")
                
                with audio_state.lock:
                    if len(raw_scores) == 1:
                        speech_prob = float(raw_scores[0])
                        if speech_prob > 0.5:
                            audio_state.last_scores = [(1, speech_prob), (0, 1.0-speech_prob)]
                            logger.info(f"🔄 Converted: class 1 (speech) = {speech_prob:.3f}, class 0 = {1.0-speech_prob:.3f}")
                        else:
                            audio_state.last_scores = [(0, 1.0-speech_prob), (1, speech_prob)]
                            logger.info(f"🔄 Converted: class 0 (no_speech) = {1.0-speech_prob:.3f}, class 1 = {speech_prob:.3f}")
                    else:
                        audio_state.last_scores = results
                    audio_state.last_inference_at = time.time()
            else:
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
        logger.info("🎙️ Background audio stopped.")

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
    logger.info("Audio thread stopped.")

# ============================================================================
# Main
# ============================================================================
def main():
    logger.info("\n" + "="*70)
    logger.info("SmartBP Pi5 - Mic + YAMNet Test (Standalone)")
    logger.info("="*70)
    
    # Load model
    try:
        backend.load()
        logger.info(f"✓ Model loaded: {backend.backend_name}")
    except Exception as e:
        logger.error(f"✗ Model load failed: {e}")
        return
    
    # Start audio capture
    try:
        start_audio_if_possible()
        logger.info("✓ Audio capture started")
    except Exception as e:
        logger.error(f"✗ Audio start failed: {e}")
        return
    
    # Run for 30 seconds
    logger.info("\nRunning for 30 seconds... Press Ctrl+C to stop\n")
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    finally:
        stop_audio()
        logger.info("\n✓ Test completed!")
        logger.info("="*70)

if __name__ == "__main__":
    main()
