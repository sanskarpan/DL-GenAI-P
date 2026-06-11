import os, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'demo_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

clf     = bundle['clf']
scaler  = bundle['scaler']
GENRES  = bundle['genres']
MODEL_F1 = bundle.get('f1', 0.54)

GENRE_EMOJI = {
    'blues':'🎸', 'classical':'🎻', 'country':'🤠', 'disco':'🪩',
    'hiphop':'🎤', 'jazz':'🎷', 'metal':'🤘', 'pop':'🎵',
    'reggae':'🌴', 'rock':'🎸'
}

# ── Audio config ───────────────────────────────────────────────────────────────
SR       = 22050
N_MFCC   = 40
N_FFT    = 2048
HOP      = 512
DURATION = 10.0

# ── Feature extraction (must match training) ───────────────────────────────────
def extract_features(audio, sr):
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    # fixed length
    target = int(DURATION * SR)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        # centre crop
        start = (len(audio) - target) // 2
        audio = audio[start:start + target]
    audio = audio.astype(np.float32)

    mfcc  = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    d1    = librosa.feature.delta(mfcc)
    d2    = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP)
    sc    = librosa.feature.spectral_centroid(y=audio, sr=SR, hop_length=HOP)
    sb    = librosa.feature.spectral_bandwidth(y=audio, sr=SR, hop_length=HOP)
    sroll = librosa.feature.spectral_rolloff(y=audio, sr=SR, hop_length=HOP)

    return np.concatenate([
        mfcc.mean(1), mfcc.std(1),
        d1.mean(1),   d1.std(1),
        d2.mean(1),   d2.std(1),
        chroma.mean(1), chroma.std(1),
        sc.mean(1),   sc.std(1),
        sb.mean(1),   sb.std(1),
        sroll.mean(1), sroll.std(1),
    ]).astype(np.float32)

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(audio_path):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    feat = extract_features(audio, sr)
    feat_s = scaler.transform(feat.reshape(1, -1))
    proba = clf.predict_proba(feat_s)[0]
    return {GENRES[i]: float(proba[i]) for i in range(len(GENRES))}

# ── Visualisation ──────────────────────────────────────────────────────────────
def make_figure(audio_path, proba_dict):
    audio, sr = librosa.load(audio_path, sr=SR, mono=True, duration=DURATION)
    target = int(DURATION * SR)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    mel  = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128, n_fft=N_FFT, hop_length=HOP)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f1a')

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                          left=0.08, right=0.96, top=0.92, bottom=0.08)

    # ── Mel spectrogram ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    img = librosa.display.specshow(mel_db, sr=SR, hop_length=HOP,
                                   x_axis='time', y_axis='mel', ax=ax1, cmap='magma')
    ax1.set_title('Log-Mel Spectrogram', color='white', fontsize=13, pad=6)
    ax1.tick_params(colors='white'); ax1.yaxis.label.set_color('white')
    ax1.xaxis.label.set_color('white')
    for spine in ax1.spines.values(): spine.set_edgecolor('#444')
    cb = fig.colorbar(img, ax=ax1, format='%+2.0f dB')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # ── MFCC heatmap ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    img2 = librosa.display.specshow(mfcc, sr=SR, hop_length=HOP,
                                    x_axis='time', ax=ax2, cmap='coolwarm')
    ax2.set_title('MFCC Coefficients', color='white', fontsize=11, pad=4)
    ax2.set_ylabel('MFCC #', color='white'); ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    for spine in ax2.spines.values(): spine.set_edgecolor('#444')

    # ── Waveform ──────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    times = np.linspace(0, DURATION, len(audio))
    ax3.plot(times, audio, color='#00d4ff', alpha=0.8, linewidth=0.6)
    ax3.fill_between(times, audio, alpha=0.2, color='#00d4ff')
    ax3.set_title('Waveform', color='white', fontsize=11, pad=4)
    ax3.set_xlabel('Time (s)', color='white'); ax3.set_ylabel('Amplitude', color='white')
    ax3.tick_params(colors='white'); ax3.set_facecolor('#1a1a2e')
    for spine in ax3.spines.values(): spine.set_edgecolor('#444')

    # ── Confidence bar chart ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    sorted_items = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
    genres_sorted = [g for g, _ in sorted_items]
    probs_sorted  = [p for _, p in sorted_items]

    bar_colors = ['#00d4ff' if i == 0 else '#4a90e2' for i in range(len(genres_sorted))]
    bars = ax4.barh([f"{GENRE_EMOJI.get(g,'🎵')} {g.capitalize()}" for g in genres_sorted],
                    probs_sorted, color=bar_colors, edgecolor='none', height=0.65)
    ax4.set_xlim(0, 1.0)
    ax4.set_xlabel('Confidence', color='white', fontsize=11)
    ax4.set_title('Genre Prediction Confidence', color='white', fontsize=12, pad=6)
    ax4.tick_params(colors='white'); ax4.set_facecolor('#1a1a2e')
    for spine in ax4.spines.values(): spine.set_edgecolor('#444')
    for bar, val in zip(bars, probs_sorted):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val*100:.1f}%', va='center', color='white', fontsize=9)

    fig.set_facecolor('#0f0f1a')
    return fig

# ── Main inference function ────────────────────────────────────────────────────
def classify_audio(audio_path):
    if audio_path is None:
        return None, "Please upload an audio file."

    try:
        proba_dict = predict(audio_path)
        top_genre  = max(proba_dict, key=proba_dict.get)
        top_conf   = proba_dict[top_genre]

        fig = make_figure(audio_path, proba_dict)

        result_md = (
            f"## {GENRE_EMOJI.get(top_genre,'🎵')} Predicted Genre: **{top_genre.upper()}**\n\n"
            f"**Confidence:** {top_conf*100:.1f}%\n\n"
            f"**Runner-up:** {sorted(proba_dict.items(), key=lambda x:-x[1])[1][0].capitalize()} "
            f"({sorted(proba_dict.items(), key=lambda x:-x[1])[1][1]*100:.1f}%)\n\n"
            f"---\n*Model: MFCC + XGBoost | Feature dim: 288 | Val Macro F1: {MODEL_F1:.3f}*"
        )
        return fig, result_md

    except Exception as e:
        return None, f"Error processing audio: {str(e)}"

# ── Gradio UI ──────────────────────────────────────────────────────────────────
css = """
body { background: #0f0f1a; }
.gradio-container { background: #0f0f1a !important; color: white !important; }
h1, h2, h3 { color: #00d4ff !important; }
.gr-button-primary { background: #00d4ff !important; color: #000 !important; font-weight: bold; }
"""

with gr.Blocks(title="🎵 Audio Genre Classifier", css=css, theme=gr.themes.Base()) as demo:
    gr.Markdown("""
    # 🎵 Audio Genre Classifier
    ### IIT Madras — DL & GenAI Project | Final Score: 0.93 Macro F1

    Upload any music audio file to classify it into one of 10 genres:
    **Blues · Classical · Country · Disco · Hip-Hop · Jazz · Metal · Pop · Reggae · Rock**

    *Model: MFCC features (288-dim) + XGBoost | Pipeline matches competition notebook*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath",
                label="Upload Music File (MP3, WAV, FLAC, OGG)",
            )
            classify_btn = gr.Button("🎯 Classify Genre", variant="primary")
            result_md = gr.Markdown(
                value="*Upload a file and click Classify.*",
                min_height=200,
            )

        with gr.Column(scale=2):
            output_plot = gr.Plot(label="Audio Analysis")

    classify_btn.click(
        fn=classify_audio,
        inputs=[audio_input],
        outputs=[output_plot, result_md],
    )

    gr.Markdown("""
    ---
    ### How it works
    1. **Feature Extraction**: Extracts 288-dim feature vector — MFCC (40 coefficients × mean+std+Δ+Δ²), Chroma STFT (12 bins), Spectral features (centroid, bandwidth, rolloff)
    2. **Normalisation**: StandardScaler (zero mean, unit variance)
    3. **Classification**: XGBoost (200 trees, max_depth=6)
    4. **Visualisation**: Log-mel spectrogram, MFCC heatmap, waveform, confidence chart

    > **Best model** (0.93 F1) uses a fine-tuned AST Transformer (86.2M params) — this demo uses the lightweight MFCC pipeline for fast CPU inference.
    """)

if __name__ == "__main__":
    demo.launch()
