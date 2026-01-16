# app.py
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from src.streams.demo_stream import DemoRadarStream
from src.signal_processing import (
    compute_microdoppler_spectrogram_like_physical,
    db_to_jet_rgb_uint8,
)
from src.storage import make_sample_path, save_axis_free_png
from src.virtual_robot import VirtualVexRobot


# -----------------------------
# Helpers (Streamlit 1.36-safe)
# -----------------------------
def show_image(placeholder, img_rgb_u8):
    placeholder.image(img_rgb_u8, channels="RGB", use_column_width=True)


def load_labels_txt(path: Path):
    """
    labels.txt:
      0 push
      1 right
      2 up
    -> {0:"push", 1:"right", 2:"up"}
    """
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, label = line.split(maxsplit=1)
            mapping[int(idx)] = label.strip()
    return mapping


def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img -= img.min()
        img = (255 * img / (img.max() + 1e-9)).clip(0, 255).astype(np.uint8)
    return img


def preprocess_for_tm(img_rgb_u8: np.ndarray) -> np.ndarray:
    x = tf.convert_to_tensor(img_rgb_u8, dtype=tf.float32)
    x = tf.image.resize(x, (224, 224))
    x = x / 255.0
    return tf.expand_dims(x, axis=0)


def predict(model, labels_map, img_rgb_u8):
    x = preprocess_for_tm(img_rgb_u8)
    out = model(x, training=False).numpy().reshape(-1)

    # If not already probabilities, softmax it
    if not np.isclose(out.sum(), 1.0, atol=1e-2) or np.any(out < 0) or np.any(out > 1):
        probs = tf.nn.softmax(out).numpy()
    else:
        probs = out

    pred_idx = int(np.argmax(probs))
    pred_label = labels_map.get(pred_idx, str(pred_idx))
    conf = float(probs[pred_idx])
    probs_dict = {labels_map.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    return pred_label, conf, probs_dict


def is_windows() -> bool:
    return sys.platform.startswith("win")


# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Radar Streamlit Demo", layout="wide")
st.title("Radar Streamlit Demo – Sprint 3.0")
st.caption("Demo micro-Doppler → Teachable Machine inference → Virtual VEX robot | + Real Radar tab (Windows-only scaffold)")


# -----------------------------
# Session state
# -----------------------------
if "demo_stream" not in st.session_state:
    st.session_state.demo_stream = DemoRadarStream(fs_hz=2000)

if "robot" not in st.session_state:
    st.session_state.robot = VirtualVexRobot()

if "model" not in st.session_state:
    st.session_state.model = None

if "labels_map" not in st.session_state:
    st.session_state.labels_map = None

if "last_img" not in st.session_state:
    st.session_state.last_img = None

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None  # (label, conf, probs)

if "running" not in st.session_state:
    st.session_state.running = False


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

data_source = st.sidebar.radio(
    "Data Source",
    ["Demo (Synthetic)", "Real Radar (Windows)"],
    index=0,
)

st.sidebar.divider()

demo_mode = st.sidebar.selectbox(
    "Gesture (demo stream)",
    ["pushing", "up", "right"],
    index=0,
)

chunk_size = st.sidebar.slider("Chunk size (samples)", 1024, 12000, 6000, 512)
prf_hz = st.sidebar.slider("PRF (Hz)", 500, 5000, 2000, 100)

st.sidebar.divider()
st.sidebar.subheader("Live")

c_live1, c_live2 = st.sidebar.columns(2)
if c_live1.button("Start Live"):
    st.session_state.running = True
if c_live2.button("Stop Live"):
    st.session_state.running = False

auto_infer = st.sidebar.checkbox("Auto-infer while Live", value=True)
confidence_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.70, 0.01)

st.sidebar.divider()
st.sidebar.subheader("Model")

model_path = Path(st.sidebar.text_input("keras_model.h5 path", "models/keras_model.h5"))
labels_path = Path(st.sidebar.text_input("labels.txt path", "models/labels.txt"))

if st.sidebar.button("Load Model"):
    if not model_path.exists():
        st.sidebar.error(f"Missing model: {model_path.as_posix()}")
    elif not labels_path.exists():
        st.sidebar.error(f"Missing labels: {labels_path.as_posix()}")
    else:
        st.session_state.model = tf.keras.models.load_model(model_path.as_posix(), compile=False)
        st.session_state.labels_map = load_labels_txt(labels_path)
        st.sidebar.success("Model + labels loaded.")

st.sidebar.divider()
st.sidebar.subheader("Sample Collection (optional)")

label_folder = st.sidebar.selectbox("Save label folder", ["pushing", "up", "right"], index=0)
num_samples = st.sidebar.number_input("Num samples", 1, 500, 25, 1)
capture_interval_s = st.sidebar.number_input("Seconds between captures", 0.1, 10.0, 0.8, 0.1)

base_dir = Path("data")
st.sidebar.caption(f"Saving to: {base_dir}/{label_folder}/###.png")

c_cap1, c_cap2 = st.sidebar.columns(2)
capture_one = c_cap1.button("Capture 1")
capture_batch = c_cap2.button("Capture batch")


# -----------------------------
# Layout (Tabs)
# -----------------------------
tab_demo, tab_real = st.tabs(["Demo + Virtual Robot", "Real Radar (Windows)"])

# =========================================================
# TAB 1: DEMO + VIRTUAL ROBOT (your working flow)
# =========================================================
with tab_demo:
    col1, col2, col3 = st.columns([1.3, 1.0, 1.0], gap="large")

    with col1:
        st.subheader("Spectrogram")
        live_img = st.empty()
        live_info = st.empty()

    with col2:
        st.subheader("Inference")
        infer_img = st.empty()
        pred_box = st.empty()
        infer_once = st.button("Infer once (current frame)")

    with col3:
        st.subheader("Virtual Robot")
        robot_box = st.empty()
        st.markdown("**Manual Controls**")
        r1, r2, r3 = st.columns(3)
        if r1.button("Push"):
            st.session_state.robot.apply_gesture("push")
        if r2.button("Up"):
            st.session_state.robot.apply_gesture("up")
        if r3.button("Right"):
            st.session_state.robot.apply_gesture("right")

        r4, r5 = st.columns(2)
        if r4.button("Reset"):
            st.session_state.robot.reset()
        if r5.button("Undo"):
            st.session_state.robot.undo_last()

    def get_demo_frame_rgb():
        raw = st.session_state.demo_stream.read_chunk(int(chunk_size), mode=demo_mode)
        spec_db = compute_microdoppler_spectrogram_like_physical(
            raw,
            prf_hz=float(prf_hz),
            window=256,
            noverlap=200,
            nfft=1024,
        )
        rgb = db_to_jet_rgb_uint8(spec_db, vmin_db=-20, vmax_db=0)
        return ensure_rgb_uint8(rgb)

    def maybe_infer_and_drive(rgb):
        if st.session_state.model is None or st.session_state.labels_map is None:
            pred_box.info("Load the model + labels to run inference.")
            return

        label, conf, probs = predict(st.session_state.model, st.session_state.labels_map, rgb)
        st.session_state.last_pred = (label, conf, probs)

        lines = [f"Prediction: {label}", f"Confidence: {conf:.3f}", "", "Probs:"]
        for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {k}: {v:.3f}")
        pred_box.text("\n".join(lines))

        if conf >= float(confidence_thresh):
            st.session_state.robot.apply_gesture(label)

    def save_sample(rgb):
        out_path = make_sample_path(base_dir, label=label_folder, index=None)
        save_axis_free_png(rgb, out_path)
        st.sidebar.success(f"Saved: {out_path.as_posix()}")

    # Capture actions
    if capture_one:
        rgb = get_demo_frame_rgb()
        save_sample(rgb)

    if capture_batch:
        st.sidebar.info(f"Capturing {int(num_samples)} samples into '{label_folder}'...")
        for _ in range(int(num_samples)):
            rgb = get_demo_frame_rgb()
            save_sample(rgb)
            time.sleep(float(capture_interval_s))
        st.sidebar.success("Done.")

    # Main render
    rgb = get_demo_frame_rgb()
    st.session_state.last_img = rgb

    show_image(live_img, rgb)
    show_image(infer_img, rgb)

    live_info.write(
        f"Source: `{data_source}` | Mode: `{demo_mode}` | chunk: `{chunk_size}` | PRF: `{prf_hz}` | Live: `{st.session_state.running}`"
    )

    # Manual infer button
    if infer_once:
        maybe_infer_and_drive(rgb)
    else:
        if st.session_state.last_pred is None:
            pred_box.info("No inference yet. Click 'Infer once' or enable Auto-infer.")
        else:
            label, conf, _ = st.session_state.last_pred
            pred_box.text(f"Prediction: {label}\nConfidence: {conf:.3f}")

    # Auto inference during live
    if st.session_state.running and auto_infer:
        maybe_infer_and_drive(rgb)

    robot_box.text(st.session_state.robot.render_text())

    # Live loop rerun (only when running)
    if st.session_state.running:
        time.sleep(0.25)
        st.rerun()


# =========================================================
# TAB 2: REAL RADAR (SPRINT 3.0 SCAFFOLD)
# =========================================================
with tab_real:
    st.subheader("Real Radar (BGT60TR13C) – Sprint 3.0")
    st.write(
        "This tab is the **safe scaffold**. On macOS it will never import `ifxradarsdk`.\n\n"
        "In **Sprint 3.1**, we’ll wire your working `InfineoenManager.py + data_collection.py + processing_utils.py` "
        "to show the true physical-style spectrogram live."
    )

    if not is_windows():
        st.warning("Real Radar mode is **Windows-only** (your BGT60TR13C SDK + USB pipeline). Switch back to Demo on macOS.")
    else:
        st.info("You’re on Windows — next sprint we’ll add Connect → Fetch frames → Live spectrogram here.")
        st.markdown(
            "**Sprint 3.1 plan:**\n"
            "- Lazy-import `ifxradarsdk` (so the app still runs without radar)\n"
            "- Load `cfg_simo_chirp.json` + `cfg_simo_seq.json`\n"
            "- Call your `radar.fetch_n_frames(n_frames=20)`\n"
            "- Use your `processing_utils.spectrogram(...)` (or equivalent) to match physical images\n"
        )
        st.code(
            "Tip: keep the real-radar code in src/streams/real_stream.py\n"
            "so demo stays clean and cross-platform."
        )