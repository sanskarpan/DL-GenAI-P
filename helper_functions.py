import torch
import numpy as np
import random
import torchaudio
import os
import glob
from pathlib import Path

# --- SET YOUR KAGGLE PATHS ---
INPUT_BASE = '/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup'
WORKING_BASE = '/kaggle/working'

STEMS_PATH = os.path.join(INPUT_BASE, 'genres_stems')
NOISE_PATH = os.path.join(INPUT_BASE, 'ESC-50-master/audio')
OUTPUT_PATH = os.path.join(WORKING_BASE, 'synthetic_mashups/train')


def seed_everything(seed=42):
    """Locks all random seeds for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Forces deterministic algorithms
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

# Execute immediately at the top of the script
seed_everything(42)







def generate_synthetic_dataset(stems_dir, noise_dir, output_dir, samples_per_genre=50, target_sr=22050, duration=30):
    """Generates deterministic noisy mashups and saves them to /kaggle/working/."""
    genres = ["blues", "classical", "country", "disco", "hiphop",
"jazz", "metal", "pop", "reggae", "rock"
]
    target_length = target_sr * duration
    
    # Get noise files from read-only input
    noise_files = glob.glob(os.path.join(noise_dir, '**', '*.wav'), recursive=True)
    
    for genre in genres:
        # Create output directories in the writable /kaggle/working/ directory
        genre_out_dir = Path(output_dir) / genre
        genre_out_dir.mkdir(parents=True, exist_ok=True)
        
        song_folders = glob.glob(os.path.join(stems_dir, genre, '*'))
        if not song_folders: 
            print(f"Warning: No songs found for genre {genre}")
            continue
        
        for i in range(samples_per_genre):
            chosen_songs = random.sample(song_folders, 4)
            stems = []
            stem_types = ['drums.wav', 'vocals.wav', 'bass.wav', 'other.wav']
            
            for song, stem_type in zip(chosen_songs, stem_types):
                stem_path = os.path.join(song, stem_type)
                if os.path.exists(stem_path):
                    waveform, sr = torchaudio.load(stem_path)
                    
                    # Basic Resampling check (if needed)
                    if sr != target_sr:
                        resampler = torchaudio.transforms.Resample(sr, target_sr)
                        waveform = resampler(waveform)

                    if waveform.shape[1] > target_length:
                        waveform = waveform[:, :target_length]
                    elif waveform.shape[1] < target_length:
                        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
                    stems.append(waveform)
            
            if len(stems) == 4:
                mashup = torch.stack(stems).sum(dim=0)
                mashup = mashup / (torch.max(torch.abs(mashup)) + 1e-8)
                
                noise_file = random.choice(noise_files)
                noise, _ = torchaudio.load(noise_file)
                
                if noise.shape[1] > target_length:
                    noise = noise[:, :target_length]
                    
                start_idx = random.randint(0, target_length - noise.shape[1])
                intensity = random.uniform(0.1, 0.4)
                
                mashup[:, start_idx:start_idx + noise.shape[1]] += (noise * intensity)
                mashup = mashup / (torch.max(torch.abs(mashup)) + 1e-8)
                
                # Save to /kaggle/working/
                out_path = genre_out_dir / f"mashup_{i:03d}.wav"
                torchaudio.save(str(out_path), mashup, target_sr)

# Run the generation
generate_synthetic_dataset(STEMS_PATH, NOISE_PATH, OUTPUT_PATH, samples_per_genre=50)




import os
import glob
import torch
import torchaudio
from pathlib import Path

def extract_and_save_features(input_dir, output_dir, target_sr=22050):
    """Converts audio to Mel-spectrograms in dB and saves as PyTorch tensors."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=2048, hop_length=512, n_mels=128
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    # Find all .wav files in the input directory
    wav_files = glob.glob(os.path.join(input_dir, '**', '*.wav'), recursive=True)
    
    if not wav_files:
        print(f"Warning: No .wav files found in {input_dir}")
        return

    for wav_path in wav_files:
        # Replicate directory structure
        rel_path = os.path.relpath(wav_path, input_dir)
        out_path = Path(output_dir) / rel_path
        out_path = out_path.with_suffix('.pt')
        
        # Ensure the target directory exists in /kaggle/working/
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process and save
        waveform, sr = torchaudio.load(wav_path)
        mel_spec = mel_transform(waveform)
        mel_spec_db = amplitude_to_db(mel_spec)
        
        torch.save(mel_spec_db, out_path)
    
    print(f"Successfully saved {len(wav_files)} feature files to {output_dir}")


INPUT_DIR = '/kaggle/working/synthetic_mashups/train'
OUTPUT_DIR = '/kaggle/working/features/train'

extract_and_save_features(INPUT_DIR, OUTPUT_DIR)


