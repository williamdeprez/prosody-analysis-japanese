import numpy as np
import pandas as pd
import librosa
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

audio_file = BASE_DIR / "data" / "raw_audio" / "jsut_ver1.1" / "jsut_ver1.1" / "voiceactress100" / "wav" / "VOICEACTRESS100_001.wav"
output_file = BASE_DIR / "data" / "processed" / "example_pitch.csv"


def extract_pitch(audio_path: Path, fmin: float = 75.0, fmax: float = 400.0, frame_length: int = 2048, hop_length: int = 256) -> pd.DataFrame:
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Extract pitch using probabilistic YIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Time axis
    times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sr,
        hop_length=hop_length
    )

    # Build DataFrame
    df = pd.DataFrame({
        "time": times,
        "f0": f0,
        "voiced": voiced_flag
    })

    return df


def export_pitch_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_pitch = extract_pitch(audio_file)

    export_pitch_csv(df_pitch, output_file)

    print(f"Pitch extraction complete. Saved to {output_file}")
