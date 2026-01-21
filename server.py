import streamlit as st
import numpy as np
import torch
import librosa
import pyloudnorm as pyln
from df.enhance import enhance, init_df
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import cv2
import imageio_ffmpeg
import io
import os
import sys
import math
import re
import ctypes
import subprocess
import tempfile
import shutil
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

if not torch.cuda.is_available():
    st.error("CUDA não está disponível neste ambiente. Este app foi configurado para rodar somente em GPU (CUDA).")
    st.stop()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = "cuda"
TARGET_LUFS_DEFAULT = -16.0
DEMUCS_MODEL = "htdemucs"
DEMUCS_SEGMENT = 6
DEMUCS_SHIFTS = 0
DENOISE_VOCALS_DEFAULT = 0.6
LFE_AMOUNT_DEFAULT = 0.4
REAR_AMOUNT_DEFAULT = 0.65
DEMUCS_SR = 44100

# =============================
# UI CONFIG
# =============================
st.set_page_config(
    page_title="Deep Audio Cleaner",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
body { background-color: #0e1117; color: #fafafa; }
</style>
""", unsafe_allow_html=True)

st.title("🎧 Deep Audio Cleaner")
try:
    cuda_name = str(torch.cuda.get_device_name(0))
except Exception:
    cuda_name = "GPU CUDA"
st.caption(f"CUDA: {cuda_name}")

input_mode = st.radio("Entrada", ["Vídeo", "Áudio"], horizontal=True)
has_video_input = input_mode == "Vídeo"

autopan = st.sidebar.toggle(
    "Auto-pan por movimento do vídeo",
    value=True,
    disabled=not has_video_input,
)
autopan_amount = st.sidebar.slider(
    "Intensidade do auto-pan",
    0.0,
    2.0,
    1.0,
    0.05,
    disabled=not (has_video_input and autopan),
)

TARGET_LUFS = st.sidebar.slider(
    "Normalização (LUFS)",
    -30.0,
    -8.0,
    TARGET_LUFS_DEFAULT,
    1.0,
)

DENOISE_VOCALS = st.sidebar.slider(
    "Força do DeepFilter na voz",
    0.0,
    1.0,
    DENOISE_VOCALS_DEFAULT,
    0.05,
)

LFE_AMOUNT = st.sidebar.slider(
    "Intensidade do LFE (subwoofer)",
    0.0,
    1.0,
    LFE_AMOUNT_DEFAULT,
    0.05,
)

REAR_AMOUNT = st.sidebar.slider(
    "Intensidade dos canais traseiros",
    0.0,
    1.0,
    REAR_AMOUNT_DEFAULT,
    0.05,
)

# =============================
# LOAD MODEL (CACHE)
# =============================
@st.cache_resource
def load_df(device: str):
    model, state, suffix = init_df()
    try:
        model = model.to(device)
    except Exception:
        pass
    return model, state, suffix

df_model, df_state, _ = load_df(DEVICE)

# =============================
# FILE UPLOAD (VIDEO/ÁUDIO)
# =============================
uploaded_video = None
uploaded_wav = None
if has_video_input:
    uploaded_video = st.file_uploader("Arraste um vídeo (MP4/MOV/MKV)", type=["mp4", "mov", "mkv", "webm"])
    if uploaded_video is None:
        st.stop()
else:
    uploaded_wav = st.file_uploader("Arraste um áudio WAV", type=["wav"])
    if uploaded_wav is None:
        st.stop()

def _pick_device(choice: str) -> str:
    if choice == "CPU":
        return "cpu"
    if choice.startswith("CUDA"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _quality_params(quality: str) -> tuple[int, int]:
    if quality == "Rápido":
        return 6, 0
    if quality == "Melhor":
        return 7, 1
    return 7, 1

def _run_ffmpeg_extract_wav(video_path: str, wav_path: str, sr: int):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(int(sr)),
        "-ac", "2",
        wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _read_wav(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    return audio, sr

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    if audio.ndim == 1:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    chans = []
    for ch in range(audio.shape[1]):
        chans.append(librosa.resample(audio[:, ch], orig_sr=orig_sr, target_sr=target_sr).astype(np.float32))
    n = min(c.shape[0] for c in chans) if chans else 0
    return np.stack([c[:n] for c in chans], axis=1)

def _write_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def _write_flac_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()

def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio

def _reorder_channels(x: np.ndarray, order: list[int]) -> np.ndarray:
    if x.ndim != 2:
        return x
    if x.shape[1] != len(order):
        return x
    return x[:, np.array(order, dtype=np.int64)]

def _delay_samples(x: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return x
    n = int(x.shape[0])
    if delay >= n:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[delay:] = x[: n - delay]
    return out

@st.cache_resource
def _ffmpeg_encoders() -> set[str]:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        proc = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out = proc.stdout or ""
    except Exception:
        out = ""
    enc = set()
    for line in out.splitlines():
        m = re.match(r"^\s*[A-Z\.]{6}\s+([0-9A-Za-z_]+)\s", line)
        if m:
            enc.add(m.group(1))
    return enc

def _ffmpeg_encode_audio_bytes(audio: np.ndarray, sr: int, codec: str, out_ext: str) -> bytes | None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_dir = tempfile.mkdtemp(prefix="deep_audio_encode_")
    in_wav = os.path.join(tmp_dir, "in.wav")
    out_path = os.path.join(tmp_dir, f"out{out_ext}")
    sf.write(in_wav, audio, sr, format="WAV", subtype="PCM_16")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        in_wav,
        "-c:a",
        codec,
        out_path,
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.exists(out_path):
        return None
    with open(out_path, "rb") as f:
        return f.read()

def _integrated_lufs(audio: np.ndarray, sr: int) -> float:
    downmix = audio.mean(axis=1) if audio.ndim == 2 else audio
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(downmix)

def _normalize_to_lufs(audio: np.ndarray, sr: int, target: float) -> np.ndarray:
    lufs = _integrated_lufs(audio, sr)
    if not np.isfinite(lufs):
        return audio
    gain = 10 ** ((target - lufs) / 20.0)
    return audio * gain

def _voice_normalize_10s(
    vocals: np.ndarray,
    sr: int,
    target_rms_dbfs: float,
    window_sec: float = 10.0,
) -> np.ndarray:
    if vocals.ndim != 2 or vocals.shape[0] < 1:
        return vocals
    win = int(round(float(window_sec) * float(sr)))
    if win <= 0:
        return vocals

    out = vocals.astype(np.float32, copy=True)
    mono = out.mean(axis=1).astype(np.float32, copy=False)

    eps = 1e-12
    max_boost_db = 12.0
    max_cut_db = 24.0
    smooth = 0.55
    last_gain = 1.0
    prev_dbs: list[float] = []

    n = int(out.shape[0])
    for start in range(0, n, win):
        end = min(n, start + win)
        seg_m = mono[start:end]
        rms = float(np.sqrt(np.mean(seg_m * seg_m) + eps))
        curr_db = float(20.0 * np.log10(rms + eps))

        if prev_dbs:
            prev_mean = float(np.mean(prev_dbs[-3:]))
        else:
            prev_mean = curr_db

        ref_db = 0.85 * float(target_rms_dbfs) + 0.15 * prev_mean
        gain_db = float(np.clip(ref_db - curr_db, -max_cut_db, max_boost_db))
        gain = float(10.0 ** (gain_db / 20.0))
        gain = smooth * last_gain + (1.0 - smooth) * gain

        seg = out[start:end] * gain
        peak = float(np.max(np.abs(seg))) if seg.size else 0.0
        if peak > 0.99:
            seg = seg * (0.99 / peak)
            gain = gain * (0.99 / peak)

        out[start:end] = seg
        last_gain = gain
        prev_dbs.append(curr_db)

    return out.astype(np.float32, copy=False)

def _lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return np.zeros_like(x)
    sos = butter(6, cutoff_hz / (sr / 2.0), btype="lowpass", output="sos")
    if x.ndim == 1:
        return sosfiltfilt(sos, x)
    return np.stack([sosfiltfilt(sos, x[:, ch]) for ch in range(x.shape[1])], axis=1)

def _to_short_path(p: str) -> str:
    try:
        buf = ctypes.create_unicode_buffer(32768)
        r = ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, len(buf))
        if r and r < len(buf):
            return buf.value
    except Exception:
        pass
    return p

def _load_cascade(filename: str) -> cv2.CascadeClassifier | None:
    candidates: list[str] = []
    base = getattr(cv2.data, "haarcascades", "")
    if base:
        candidates.append(os.path.join(base, filename))
    cv2_dir = os.path.dirname(getattr(cv2, "__file__", "") or "")
    if cv2_dir:
        candidates.append(os.path.join(cv2_dir, "data", filename))
        candidates.append(os.path.join(cv2_dir, filename))

    for p in candidates:
        if p and os.path.exists(p):
            cc = cv2.CascadeClassifier(_to_short_path(p))
            if not cc.empty():
                return cc
    return None

def _estimate_speaker_pan(video_path: str, sample_fps: int = 8) -> tuple[np.ndarray, np.ndarray]:
    face_cascade = _load_cascade("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(_to_short_path(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and np.isfinite(fps) and fps > 1e-3 else 30.0
    step = max(1, int(round(fps / sample_fps)))

    t = []
    x = []
    idx = 0
    last = 0.0
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            val = last
            if face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    faces = face_cascade.detectMultiScale(
                        gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                    )
                    if len(faces) > 0:
                        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                        fx, fy, fw, fh = int(fx * 2), int(fy * 2), int(fw * 2), int(fh * 2)
                        cx = (fx + 0.5 * fw) / float(gray.shape[1])
                        val = float(cx * 2.0 - 1.0)
                        last = val
                except cv2.error:
                    val = last
            t.append(idx / fps)
            x.append(float(np.clip(val, -1.0, 1.0)))
        idx += 1
        ok, frame = cap.read()
    cap.release()
    if len(t) < 2:
        return np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)
    return np.array(t, dtype=np.float32), np.array(x, dtype=np.float32)

def _estimate_background_pan(video_path: str, grid: int = 3) -> tuple[np.ndarray, np.ndarray]:
    t, E, x_centers = _estimate_motion_grid(video_path, grid=grid)
    if E.shape[1] < 2:
        return np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)
    xs = []
    for j in range(E.shape[1]):
        w = E[:, j].astype(np.float32, copy=False)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 1e-8:
            xs.append(0.0)
        else:
            xs.append(float((w @ x_centers) / s))
    return t.astype(np.float32), np.clip(np.array(xs, dtype=np.float32), -1.0, 1.0)

def _apply_stereo_pan(stem: np.ndarray, pan_curve: np.ndarray, amount: float) -> np.ndarray:
    pan = np.asarray(pan_curve, dtype=np.float32).reshape(-1)
    n = int(stem.shape[0])
    m = int(pan.shape[0])
    if m != n:
        if m <= 0:
            pan = np.zeros((n,), dtype=np.float32)
        elif m > n:
            pan = pan[:n]
        else:
            pan2 = np.empty((n,), dtype=np.float32)
            pan2[:m] = pan
            pan2[m:] = float(pan[m - 1])
            pan = pan2
    pan = np.clip(pan * float(amount), -1.0, 1.0)
    if stem.ndim == 1:
        mm = stem.astype(np.float32, copy=False)
    else:
        mm = stem.mean(axis=1).astype(np.float32, copy=False)
    theta = (pan + 1.0) * (np.pi / 4.0)
    gl = np.cos(theta).astype(np.float32)
    gr = np.sin(theta).astype(np.float32)
    return np.stack([mm * gl, mm * gr], axis=1)

def _estimate_mouth_activity(video_path: str, sample_fps: int = 8) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and np.isfinite(fps) and fps > 1e-3 else 30.0
    step = max(1, int(round(fps / sample_fps)))

    face_cascade = _load_cascade("haarcascade_frontalface_default.xml")

    t = []
    v = []
    idx = 0
    prev_roi = None
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = None
            if face_cascade is not None:
                try:
                    gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    faces = face_cascade.detectMultiScale(
                        gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                    )
                    if len(faces) > 0:
                        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                        x, y, w, h = int(x * 2), int(y * 2), int(w * 2), int(h * 2)
                        x0 = max(0, x + int(0.2 * w))
                        x1 = min(gray.shape[1], x + int(0.8 * w))
                        y0 = max(0, y + int(0.6 * h))
                        y1 = min(gray.shape[0], y + int(0.95 * h))
                        roi = gray[y0:y1, x0:x1]
                except cv2.error:
                    roi = None

            if roi is None or roi.size == 0:
                h, w = gray.shape[:2]
                x0 = int(w * 0.30)
                x1 = int(w * 0.70)
                y0 = int(h * 0.55)
                y1 = int(h * 0.95)
                roi = gray[y0:y1, x0:x1]

            if roi.size == 0:
                val = 0.0
                prev_roi = None
            else:
                roi = cv2.resize(roi, (96, 64), interpolation=cv2.INTER_AREA)
                roi = roi.astype(np.float32) / 255.0
                if prev_roi is None:
                    val = 0.0
                else:
                    val = float(np.mean(np.abs(roi - prev_roi)))
                prev_roi = roi

            t.append(idx / fps)
            v.append(val)

        idx += 1
        ok, frame = cap.read()
    cap.release()

    if len(t) < 2:
        return np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)
    vv = np.array(v, dtype=np.float32)
    p10, p95 = np.percentile(vv, [10, 95])
    vv = (vv - p10) / max(p95 - p10, 1e-6)
    vv = np.clip(vv, 0.0, 1.0)
    return np.array(t, dtype=np.float32), vv

def _estimate_motion_pan(video_path: str, sample_fps: int = 6, max_frames: int = 800) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and np.isfinite(fps) and fps > 1e-3 else 30.0
    step = max(1, int(round(fps / sample_fps)))

    t = []
    v = []
    prev = None
    idx = 0
    used = 0
    ok, frame = cap.read()
    while ok and used < max_frames:
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
                fx = flow[..., 0]
                val = float(np.mean(fx))
                t.append(idx / fps)
                v.append(val)
                used += 1
            prev = gray
        idx += 1
        ok, frame = cap.read()
    cap.release()
    if len(t) < 2:
        return np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)
    vv = np.array(v, dtype=np.float32)
    vv = vv / max(np.percentile(np.abs(vv), 90), 1e-6)
    vv = np.clip(vv, -1.0, 1.0)
    return np.array(t, dtype=np.float32), vv

def _estimate_motion_grid(video_path: str, grid: int = 3, sample_fps: int = 6, max_frames: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and np.isfinite(fps) and fps > 1e-3 else 30.0
    step = max(1, int(round(fps / sample_fps)))

    t = []
    energies = []
    prev = None
    idx = 0
    used = 0
    ok, frame = cap.read()
    while ok and used < max_frames:
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (192, 108), interpolation=cv2.INTER_AREA)
            if prev is not None:
                diff = cv2.absdiff(prev, gray).astype(np.float32) / 255.0
                h, w = diff.shape
                cell_h = h // grid
                cell_w = w // grid
                e = []
                for gy in range(grid):
                    for gx in range(grid):
                        y0 = gy * cell_h
                        y1 = h if gy == grid - 1 else (gy + 1) * cell_h
                        x0 = gx * cell_w
                        x1 = w if gx == grid - 1 else (gx + 1) * cell_w
                        e.append(float(np.mean(diff[y0:y1, x0:x1])))
                energies.append(e)
                t.append(idx / fps)
                used += 1
            prev = gray
        idx += 1
        ok, frame = cap.read()
    cap.release()

    if len(t) < 2:
        gx = np.array([0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32), np.zeros((grid * grid, 2), dtype=np.float32), gx

    E = np.array(energies, dtype=np.float32).T
    x_centers = []
    for gy in range(grid):
        for gx in range(grid):
            x_centers.append((gx + 0.5) / grid * 2.0 - 1.0)
    return np.array(t, dtype=np.float32), E, np.array(x_centers, dtype=np.float32)

def _curve_to_samples(t: np.ndarray, v: np.ndarray, sr: int, n_samples: int) -> np.ndarray:
    times = np.arange(n_samples, dtype=np.float32) / float(sr)
    return np.interp(times, t, v).astype(np.float32)

def _apply_gain_curve(stem: np.ndarray, curve: np.ndarray) -> np.ndarray:
    if curve.ndim != 1:
        curve = np.asarray(curve).reshape(-1)
    n = int(stem.shape[0])
    m = int(curve.shape[0])
    if m != n:
        if m <= 0:
            curve = np.ones((n,), dtype=np.float32)
        elif m > n:
            curve = curve[:n]
        else:
            curve2 = np.empty((n,), dtype=np.float32)
            curve2[:m] = curve.astype(np.float32, copy=False)
            curve2[m:] = float(curve[m - 1])
            curve = curve2
    if stem.ndim == 1:
        return stem * curve
    return stem * curve[:, None]

def _demucs_separate_cli(
    model_name: str,
    device: str,
    segment: int,
    shifts: int,
    in_wav_path: str,
    out_dir: str,
) -> dict[str, np.ndarray]:
    def run(seg: int) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            model_name,
            "-d",
            device,
            "--segment",
            str(int(seg)),
            "--shifts",
            str(int(shifts)),
            "--overlap",
            "0.25",
            "-o",
            out_dir,
            "--filename",
            "{stem}.{ext}",
            in_wav_path,
        ]
        return subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    proc = run(segment)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        max_prefix = "Maximum segment is:"
        if max_prefix in err:
            tail = err.split(max_prefix, 1)[1].strip()
            try:
                max_seg = float(tail.split()[0])
            except Exception:
                max_seg = None
            if max_seg is not None and max_seg > 0:
                seg2 = int(max(1, min(segment, int(math.floor(max_seg)))))
                proc2 = run(seg2)
                if proc2.returncode == 0:
                    proc = proc2
                else:
                    err2 = (proc2.stderr or proc2.stdout or "").strip()
                    raise RuntimeError(err2 or err or "Falha ao rodar Demucs")
        else:
            raise RuntimeError(err or "Falha ao rodar Demucs")

    model_dir = os.path.join(out_dir, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Diretório de saída do Demucs não encontrado: {model_dir}")

    stem_files = {}
    for root, _, files in os.walk(model_dir):
        for fn in files:
            if fn.lower().endswith(".wav"):
                stem = Path(fn).stem.lower()
                stem_files[stem] = os.path.join(root, fn)

    wanted = ["vocals", "drums", "bass", "other"]
    missing = [w for w in wanted if w not in stem_files]
    if missing:
        raise FileNotFoundError(f"Stems ausentes do Demucs: {missing}. Encontrados: {sorted(stem_files.keys())}")

    stems = {}
    ref_len = None
    for name in wanted:
        x, _sr = _read_wav(stem_files[name])
        stems[name] = x
        ref_len = x.shape[0] if ref_len is None else min(ref_len, x.shape[0])

    if ref_len is not None:
        for k in list(stems.keys()):
            stems[k] = stems[k][:ref_len]
    return stems

def _df_denoise_mono(sig: np.ndarray) -> np.ndarray:
    audio_t = torch.from_numpy(sig).float().to(DEVICE).unsqueeze(0)
    with torch.inference_mode():
        enhanced = enhance(df_model, df_state, audio_t)
    enhanced_cpu = enhanced.detach().cpu()
    return enhanced_cpu.squeeze().numpy()

def _df_denoise_stereo(stem: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 1e-6:
        return stem
    if stem.ndim == 1:
        den = _df_denoise_mono(stem)
        return den * amount + stem * (1.0 - amount)
    left = _df_denoise_mono(stem[:, 0])
    right = _df_denoise_mono(stem[:, 1])
    den = np.stack([left, right], axis=1)
    return den * amount + stem * (1.0 - amount)

def _apply_df_to_vocals(stems: dict[str, np.ndarray], sr: int) -> None:
    if "vocals" not in stems or DENOISE_VOCALS <= 1e-6:
        return
    try:
        df_sr = int(df_state.sr())
        v = stems["vocals"]
        v48 = _resample(v, sr, df_sr)
        v48 = _df_denoise_stereo(v48, DENOISE_VOCALS)
        stems["vocals"] = _resample(v48, df_sr, sr)
    except Exception as e:
        msg = str(e)
        if "can't convert cuda:0 device type tensor to numpy" in msg:
            return
        raise

def _mix_stems(stems: dict[str, np.ndarray], gains: dict[str, float]) -> np.ndarray:
    mix = None
    for name, stem in stems.items():
        g = float(gains.get(name, 1.0))
        if mix is None:
            mix = stem * g
        else:
            mix = mix + stem * g
    return mix if mix is not None else np.zeros((1, 2), dtype=np.float32)

def _to_5p1(stems: dict[str, np.ndarray], sr: int, pan_curves: dict[str, np.ndarray] | None, lfe_amount: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = next(iter(stems.values())).shape[0]
    out = np.zeros((n, 6), dtype=np.float32)
    stem_5p1 = {}

    def equal_power_lr(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.clip(p, -1.0, 1.0)
        theta = (p + 1.0) * (np.pi / 4.0)
        gl = np.cos(theta)
        gr = np.sin(theta)
        return gl.astype(np.float32), gr.astype(np.float32)

    def mono(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=1) if x.ndim == 2 else x

    def add_pair(ch_l: int, ch_r: int, x: np.ndarray, pan: np.ndarray | None, amount: float, label: str):
        mm = mono(x)
        if pan is None:
            gl = np.full(n, 1 / np.sqrt(2), dtype=np.float32)
            gr = np.full(n, 1 / np.sqrt(2), dtype=np.float32)
        else:
            gl, gr = equal_power_lr(pan * amount)
        y = np.zeros((n, 6), dtype=np.float32)
        y[:, ch_l] = mm * gl
        y[:, ch_r] = mm * gr
        stem_5p1[label] = y
        out[:] += y

    def add_center(x: np.ndarray, label: str):
        mm = mono(x)
        y = np.zeros((n, 6), dtype=np.float32)
        y[:, 2] = mm
        stem_5p1[label] = y
        out[:] += y

    def add_lfe(x: np.ndarray, amount: float, label: str):
        if amount <= 1e-6:
            stem_5p1[label] = np.zeros((n, 6), dtype=np.float32)
            return
        mm = mono(x)
        lp = _lowpass(mm, sr, 120.0)
        y = np.zeros((n, 6), dtype=np.float32)
        y[:, 3] = lp * amount
        stem_5p1[label] = y
        out[:] += y

    if "vocals" in stems:
        add_center(stems["vocals"], "dialogue_C")

    sop_keys = [k for k in stems.keys() if k.startswith("sop_")]
    if sop_keys:
        for k in sop_keys:
            pan = None if pan_curves is None else pan_curves.get(k)
            add_pair(0, 1, stems[k], pan, 0.75, f"{k}_front")
            add_pair(4, 5, stems[k], pan, 1.0, f"{k}_surround")
    else:
        if "drums" in stems:
            pan = None if pan_curves is None else pan_curves.get("drums")
            add_pair(0, 1, stems["drums"], pan, 1.0, "fx_FLFR")
        if "bass" in stems:
            pan = None if pan_curves is None else pan_curves.get("bass")
            add_pair(0, 1, stems["bass"], pan, 0.6, "bass_FLFR")
        if "other" in stems:
            add_pair(4, 5, stems["other"], None, 0.0, "music_surround")

    if "drums" in stems and "bass" in stems:
        add_lfe(stems["drums"] + stems["bass"], lfe_amount, "lfe")
    elif "bass" in stems:
        add_lfe(stems["bass"], lfe_amount, "lfe")
    elif "drums" in stems:
        add_lfe(stems["drums"], lfe_amount, "lfe")

    return out, stem_5p1

def _to_7p1(
    stems: dict[str, np.ndarray],
    sr: int,
    pan_curves: dict[str, np.ndarray] | None,
    lfe_amount: float,
    rear_amount: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = next(iter(stems.values())).shape[0]
    out = np.zeros((n, 8), dtype=np.float32)
    stem_7p1 = {}

    def equal_power_lr(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.clip(p, -1.0, 1.0)
        theta = (p + 1.0) * (np.pi / 4.0)
        gl = np.cos(theta)
        gr = np.sin(theta)
        return gl.astype(np.float32), gr.astype(np.float32)

    def mono(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=1) if x.ndim == 2 else x

    def add_pair(ch_l: int, ch_r: int, x: np.ndarray, pan: np.ndarray | None, amount: float, label: str):
        mm = mono(x).astype(np.float32, copy=False)
        if pan is None:
            gl = np.full(n, 1 / np.sqrt(2), dtype=np.float32)
            gr = np.full(n, 1 / np.sqrt(2), dtype=np.float32)
        else:
            gl, gr = equal_power_lr(pan * amount)
        y = np.zeros((n, 8), dtype=np.float32)
        y[:, ch_l] = mm * gl
        y[:, ch_r] = mm * gr
        stem_7p1[label] = y
        out[:] += y

    def add_center(x: np.ndarray, label: str):
        mm = mono(x).astype(np.float32, copy=False)
        y = np.zeros((n, 8), dtype=np.float32)
        y[:, 2] = mm
        stem_7p1[label] = y
        out[:] += y

    def add_lfe(x: np.ndarray, amount: float, label: str):
        if amount <= 1e-6:
            stem_7p1[label] = np.zeros((n, 8), dtype=np.float32)
            return
        mm = mono(x).astype(np.float32, copy=False)
        lp = _lowpass(mm, sr, 120.0).astype(np.float32, copy=False)
        y = np.zeros((n, 8), dtype=np.float32)
        y[:, 3] = lp * amount
        stem_7p1[label] = y
        out[:] += y

    def rearize(x: np.ndarray, delay_ms: float = 12.0) -> np.ndarray:
        mm = mono(x).astype(np.float32, copy=False)
        d = int(round((delay_ms / 1000.0) * sr))
        return _delay_samples(mm, d)

    if "vocals" in stems:
        add_center(stems["vocals"], "dialogue_C")

    sop_keys = [k for k in stems.keys() if k.startswith("sop_")]
    if sop_keys:
        for k in sop_keys:
            pan = None if pan_curves is None else pan_curves.get(k)
            add_pair(0, 1, stems[k], pan, 0.75, f"{k}_front")
            add_pair(4, 5, stems[k], pan, 1.0, f"{k}_side")
            if rear_amount > 1e-6:
                add_pair(6, 7, rearize(stems[k]), pan, float(rear_amount), f"{k}_rear")
    else:
        if "drums" in stems:
            pan = None if pan_curves is None else pan_curves.get("drums")
            add_pair(0, 1, stems["drums"], pan, 1.0, "fx_FLFR")
        if "bass" in stems:
            pan = None if pan_curves is None else pan_curves.get("bass")
            add_pair(0, 1, stems["bass"], pan, 0.6, "bass_FLFR")
        if "other" in stems:
            side_sig = stems["other"]
            add_pair(4, 5, side_sig, None, 0.0, "music_side")
            if rear_amount > 1e-6:
                add_pair(6, 7, rearize(side_sig), None, float(rear_amount), "music_rear")

    if "drums" in stems and "bass" in stems:
        add_lfe(stems["drums"] + stems["bass"], lfe_amount, "lfe")
    elif "bass" in stems:
        add_lfe(stems["bass"], lfe_amount, "lfe")
    elif "drums" in stems:
        add_lfe(stems["drums"], lfe_amount, "lfe")

    return out, stem_7p1

def _channel_stems_surround(
    stems: dict[str, np.ndarray],
    sr: int,
    pan_curves: dict[str, np.ndarray] | None,
    mode: str,
    lfe_amount: float,
    rear_amount: float,
    rear_split: float,
    voice_bleed: float,
) -> dict[str, np.ndarray]:
    n = next(iter(stems.values())).shape[0]

    def mono(x: np.ndarray | None) -> np.ndarray:
        if x is None:
            return np.zeros((n,), dtype=np.float32)
        if x.ndim == 1:
            return x.astype(np.float32, copy=False)
        return x.mean(axis=1).astype(np.float32, copy=False)

    def equal_power_lr(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.clip(p.astype(np.float32, copy=False), -1.0, 1.0)
        theta = (p + 1.0) * (np.pi / 4.0)
        gl = np.cos(theta).astype(np.float32)
        gr = np.sin(theta).astype(np.float32)
        return gl, gr

    def stereo(mm: np.ndarray, pan_key: str | None = None, amount: float = 1.0) -> np.ndarray:
        if pan_curves is None or not pan_key or pan_key not in pan_curves:
            gl = np.full(n, 1 / np.sqrt(2), dtype=np.float32)
            gr = np.full(n, 1 / np.sqrt(2), dtype=np.float32)
        else:
            gl, gr = equal_power_lr(pan_curves[pan_key] * float(amount))
        return np.stack([mm * gl, mm * gr], axis=1)

    vocals = mono(stems.get("vocals"))
    drums = mono(stems.get("drums"))
    bass = mono(stems.get("bass"))
    other = mono(stems.get("other"))
    noise = mono(stems.get("noise"))

    ambient = 0.90 * drums + 0.60 * noise
    background = other + 0.25 * noise

    front_amb = stereo(ambient, pan_key="drums", amount=1.0)
    bleed = stereo(vocals * float(voice_bleed), pan_key="vocals", amount=1.0)
    front = front_amb + bleed

    c = vocals
    lfe = _lowpass(bass + drums, sr, 120.0).astype(np.float32, copy=False) * float(lfe_amount)

    out: dict[str, np.ndarray] = {}
    out["FL"] = front[:, 0]
    out["FR"] = front[:, 1]
    out["C"] = c
    out["LFE"] = lfe

    if mode == "5.1":
        s = stereo(background, pan_key="other", amount=1.0)
        out["SL"] = s[:, 0]
        out["SR"] = s[:, 1]
    else:
        rs = float(np.clip(rear_split, 0.0, 1.0))
        side_sig = background * (1.0 - rs)
        rear_sig = background * rs
        s_side = stereo(side_sig, pan_key="other", amount=1.0)
        s_rear = stereo(_delay_samples(rear_sig, int(round(sr * 0.012))), pan_key="other", amount=1.0) * float(rear_amount)
        out["SL"] = s_side[:, 0]
        out["SR"] = s_side[:, 1]
        out["RL"] = s_rear[:, 0]
        out["RR"] = s_rear[:, 1]

    for k in list(out.keys()):
        out[k] = np.clip(out[k], -1.0, 1.0).astype(np.float32, copy=False)
    return out

def _sop_separate(
    audio: np.ndarray,
    sr: int,
    n_sources: int,
    nmf_iter: int,
    video_path: str,
    grid: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raise RuntimeError("Sound-of-Pixels (SOP) foi desativado nesta versão.")

st.subheader("Escolha o que exportar")
export_mix_5p1 = st.checkbox("5.1 WAV", value=False)
export_mix_7p1 = st.checkbox("7.1 WAV", value=False)
export_mix_stereo = st.checkbox("Stereo WAV", value=True)
export_stems_stereo = st.checkbox("Stems stereo", value=False)
export_stems_5p1 = st.checkbox("Stems 5.1", value=False)

def _cleanup_tmp_dir():
    tmp_dir = st.session_state.pop("tmp_dir", None)
    if isinstance(tmp_dir, str) and tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

def _safe_zip_basename(name: str) -> str:
    base = Path(name).stem.strip() or "export"
    base = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in base])
    return base[:80] or "export"

def _generate_exports_zip() -> tuple[bytes, str]:
    _cleanup_tmp_dir()
    tmp_dir = tempfile.mkdtemp(prefix="deep_audio_cleaner_")
    try:
        wav_path = os.path.join(tmp_dir, "audio.wav")
        video_path = None
        input_name = uploaded_video.name if has_video_input else uploaded_wav.name

        if has_video_input:
            video_suffix = Path(uploaded_video.name).suffix or ".mp4"
            video_path = os.path.join(tmp_dir, f"input{video_suffix}")
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
        else:
            with open(wav_path, "wb") as f:
                f.write(uploaded_wav.getbuffer())

        if has_video_input:
            _run_ffmpeg_extract_wav(video_path, wav_path, DEMUCS_SR)
            audio, sr = _read_wav(wav_path)
        else:
            audio, sr = _read_wav(wav_path)
            audio = _ensure_stereo(audio)
            if sr != DEMUCS_SR:
                audio = _resample(audio, sr, DEMUCS_SR)
                sr = DEMUCS_SR
            sf.write(wav_path, audio, sr, format="WAV", subtype="PCM_16")

        with st.spinner("Separando camadas (Demucs em CUDA)..."):
            stems = _demucs_separate_cli(DEMUCS_MODEL, DEVICE, DEMUCS_SEGMENT, DEMUCS_SHIFTS, wav_path, tmp_dir)

        with st.spinner("Aplicando DeepFilterNet na voz (CUDA)..."):
            _apply_df_to_vocals(stems, sr)

        common_len = int(audio.shape[0])
        for s in stems.values():
            common_len = min(common_len, int(s.shape[0]))
        if common_len > 0:
            audio = audio[:common_len]
            for k in list(stems.keys()):
                stems[k] = stems[k][:common_len]

        stems_sum = None
        for s in stems.values():
            stems_sum = s if stems_sum is None else (stems_sum + s)
        noise = audio - stems_sum if stems_sum is not None else np.zeros_like(audio)
        stems["noise"] = noise.astype(np.float32, copy=False)

        stems_for_mix = {k: v for k, v in stems.items() if k in {"vocals", "drums", "bass", "other"}}
        pan_curves = None
        if has_video_input and autopan and video_path is not None:
            try:
                with st.spinner("Estimando movimento do vídeo para auto-pan..."):
                    t_pan, v_pan = _estimate_motion_pan(video_path)
                if (
                    isinstance(t_pan, np.ndarray)
                    and isinstance(v_pan, np.ndarray)
                    and t_pan.size >= 2
                    and v_pan.size >= 2
                    and common_len > 0
                ):
                    v_scaled = v_pan.astype(np.float32, copy=False) * float(autopan_amount)
                    pan_vec = _curve_to_samples(t_pan, v_scaled, sr, common_len)
                    pan_curves = {}
                    for key in ("drums", "bass", "other"):
                        if key in stems_for_mix:
                            pan_curves[key] = pan_vec
            except Exception:
                pan_curves = None

        mix_stereo = _mix_stems(stems_for_mix, {})
        mix_stereo = _normalize_to_lufs(mix_stereo, sr, TARGET_LUFS)
        mix_stereo = np.clip(mix_stereo, -1.0, 1.0).astype(np.float32, copy=False)

        base = _safe_zip_basename(input_name)
        zip_name = f"{base}_exports.zip"
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if export_mix_stereo:
                zf.writestr("mix_stereo.wav", _write_wav_bytes(mix_stereo, sr))

            if export_stems_stereo:
                name_map = {
                    "vocals": "stems_stereo/voz.wav",
                    "drums": "stems_stereo/efeitos_fundo.wav",
                    "bass": "stems_stereo/bass.wav",
                    "other": "stems_stereo/trilha_sonora.wav",
                    "noise": "stems_stereo/noise.wav",
                }
                for k, path_in_zip in name_map.items():
                    if k in stems:
                        zf.writestr(path_in_zip, _write_wav_bytes(np.clip(stems[k], -1.0, 1.0), sr))

            if export_mix_5p1 or export_stems_5p1:
                mix_5p1, stems_5p1 = _to_5p1(stems_for_mix, sr, pan_curves, float(LFE_AMOUNT))
                mix_5p1 = _normalize_to_lufs(mix_5p1, sr, TARGET_LUFS)
                mix_5p1 = np.clip(mix_5p1, -1.0, 1.0).astype(np.float32, copy=False)
                if export_mix_5p1:
                    zf.writestr("mix_5p1.wav", _write_wav_bytes(mix_5p1, sr))
                if export_stems_5p1:
                    for label, sig in stems_5p1.items():
                        zf.writestr(f"stems_5p1/{label}.wav", _write_wav_bytes(np.clip(sig, -1.0, 1.0), sr))

            if export_mix_7p1:
                mix_7p1, _stems_7p1 = _to_7p1(stems_for_mix, sr, pan_curves, float(LFE_AMOUNT), float(REAR_AMOUNT))
                mix_7p1 = _normalize_to_lufs(mix_7p1, sr, TARGET_LUFS)
                mix_7p1 = np.clip(mix_7p1, -1.0, 1.0).astype(np.float32, copy=False)
                zf.writestr("mix_7p1.wav", _write_wav_bytes(mix_7p1, sr))

        st.session_state["tmp_dir"] = tmp_dir
        return zip_buf.getvalue(), zip_name
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

def _after_download_cleanup():
    _cleanup_tmp_dir()
    st.session_state.pop("export_zip_bytes", None)
    st.session_state.pop("export_zip_name", None)

generate_clicked = st.button("Gerar ZIP")
if generate_clicked:
    if not (export_mix_5p1 or export_mix_7p1 or export_mix_stereo or export_stems_stereo or export_stems_5p1):
        st.error("Selecione pelo menos uma exportação.")
        st.stop()
    try:
        zip_bytes, zip_name = _generate_exports_zip()
    except Exception as e:
        st.error(f"Falha ao gerar exportações: {e}")
        st.stop()
    st.session_state["export_zip_bytes"] = zip_bytes
    st.session_state["export_zip_name"] = zip_name

zip_bytes = st.session_state.get("export_zip_bytes")
zip_name = st.session_state.get("export_zip_name") or "exports.zip"
if isinstance(zip_bytes, (bytes, bytearray)) and zip_bytes:
    st.download_button(
        "⬇️ Baixar ZIP",
        data=zip_bytes,
        file_name=str(zip_name),
        on_click=_after_download_cleanup,
    )
