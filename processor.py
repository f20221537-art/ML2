import os
import subprocess
import numpy as np
import librosa

# Constants matched to your notebook
TARGET_SR = 22050

def extract_mel_spectrogram(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    return librosa.power_to_db(mel_spec, ref=np.max)

def run_demucs(audio_path):
    # Calls demucs exactly like your Cell 4
    command = ["python", "-m", "demucs", "-d", "cpu", audio_path]
    subprocess.run(command, check=True)
    
    # Logic to find the output folder
    filename = os.path.basename(audio_path).replace(".wav", "")
    return os.path.join("separated", "htdemucs", filename)

def merge_logic(time_templates, max_gap, min_play):
    instrument_groups = {}
    final_results = {}
    
    for p in time_templates:
        instrument_groups.setdefault(p['instrument'], []).append((p['start'], p['end']))

    for inst, timings in instrument_groups.items():
        timings.sort(key=lambda x: x)
        blocks = []
        for start, end in timings:
            if not blocks:
                blocks.append([start, end])
            else:
                if start - blocks[-1] <= max_gap:
                    blocks[-1] = max(blocks[-1], end)
                else:
                    blocks.append([start, end])
        
        # Filter by minimum play time
        valid = [b for b in blocks if (b-b) >= min_play]
        if valid:
            final_results[inst] = valid
            
    return final_results
