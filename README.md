# Radar Streamlit Demo (Demo → Real Radar → VEX Control)

[Link to Notion](https://www.notion.so/Radar-Streamlit-Demo-2ea3e03f1fae809d9f0fec067eb1c876?source=copy_link)


Streamlit application for teaching radar-based gesture classification using:
- **Demo (Synthetic)** radar stream (works on macOS + Windows)
- **Teachable Machine (Keras)** model inference (`keras_model.h5` + `labels.txt`)
- **Virtual VEX robot** grid simulator
- **Real Radar (Windows-only)** scaffold for Infineon **BGT60TR13C** (Sprint 3.x+)

---

## Current Status (Sprint 3.0)

✅ Demo spectrogram + sample capture  
✅ Teachable Machine model loading + inference  
✅ Virtual robot moves from predictions  
🧩 Real radar tab is scaffolded (Windows-only; no SDK import on macOS)

---

## Project Structure (expected)

```text
radar_streamlit_demo/
  app.py
  src/
    __init__.py
    signal_processing.py
    storage.py
    virtual_robot.py
    streams/
      __init__.py
      demo_stream.py
      # real_stream.py  (Sprint 3.1+)
  models/
    keras_model.h5        (optional; do not commit)
    labels.txt            (optional; do not commit)
  data/                   (generated; do not commit)
  requirements.txt
  .gitignore
  README.md
