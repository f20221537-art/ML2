import streamlit as st
import os
import tensorflow as tf
import numpy as np
import librosa
from processor import extract_mel_spectrogram, run_demucs, merge_logic

st.set_page_config(page_title="AI Instrument Detector", page_icon="🎸")

# Sidebar for original notebook settings
st.sidebar.header("Model Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.10)
max_gap = st.sidebar.slider("Max Silence Gap (s)", 0.5, 5.0, 2.0)
min_play = st.sidebar.slider("Min Play Time (s)", 0.5, 5.0, 1.5)

st.title("🎵 Multi-Instrument Classifier")
st.info("This app uses the CNN model trained in 'multi_instrument_detection.ipynb'.")

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model('instrument_model.keras')
    labels = np.load('label_classes.npy', allow_pickle=True)
    return model, labels

model, label_classes = load_model_and_labels()

uploaded_file = st.file_uploader("Upload Audio (WAV)", type=["wav"])

if uploaded_file:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("temp_audio.wav")
    
    if st.button("Start AI Analysis"):
        # 1. Separation
        with st.spinner("Separating audio into stems (Demucs)..."):
            stem_folder = run_demucs("temp_audio.wav")
        
        # 2. Scanning (Cell 5 logic)
        with st.spinner("CNN scanning stems for instruments..."):
            time_templates = []
            stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
            
            for stem in stems:
                path = os.path.join(stem_folder, stem)
                if not os.path.exists(path): continue
                
                y, sr = librosa.load(path, sr=22050)
                window_size = int(1.0 * sr)
                hop_size = int(0.5 * sr)

                for start in range(0, len(y) - window_size, hop_size):
                    window = y[start : start + window_size]
                    spec = extract_mel_spectrogram(window, sr)
                    
                    if spec.shape == 44:
                        preds = model.predict(spec[np.newaxis, ..., np.newaxis], verbose=0)
                        for idx, prob in enumerate(preds):
                            if prob > conf_thresh:
                                time_templates.append({
                                    "start": start / sr,
                                    "end": (start + window_size) / sr,
                                    "instrument": label_classes[idx]
                                })

            # 3. Merge and Display
            results = merge_logic(time_templates, max_gap, min_play)
            
            st.success("Analysis Complete!")
            for inst, times in results.items():
                with st.expander(f"Instrument: {inst.upper()}"):
                    for t in times:
                        st.write(f"⏱️ {t:.1f}s — {t:.1f}s  ({t-t:.1f}s)")
