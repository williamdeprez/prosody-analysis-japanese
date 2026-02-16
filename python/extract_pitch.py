import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import matplotlib.pyplot as plt

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

def clean_pitch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["f0_interpolation"] = df["f0"].interpolate(method="linear")

    df["f0_interpolation"] = df["f0_interpolation"].bfill().ffill()

    return df

def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize time to [0, 1] range for better model training. Each sentence is treated as a separate sequence, so time is normalized within each sentence.
    
    :param df: Description
    :type df: pd.DataFrame
    :return: Description
    :rtype: DataFrame
    """
    df = df.copy()
    t_min = df["time"].min()
    t_max = df["time"].max()

    df["time_normalized"] = (df["time"] - t_min) / (t_max - t_min)
    return df

def resample_to_fixed_grid(df: pd.DataFrame, n_points: int = 100) -> pd.DataFrame:
    df = df.copy()

    grid = np.linspace(0, 1, n_points)

    f0_resampled = np.interp(
        grid,
        df["time_normalized"],
        df["f0_interpolation"]
    )

    return pd.DataFrame({
        "time_normalized": grid,
        "f0_resampled": f0_resampled
    })

def plot_pitch(df: pd.DataFrame, time_col: str, pitch_col: str, title: str = "Pitch Contour") -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(df[time_col], df[pitch_col])
    plt.xlabel("Normalized Time")
    plt.ylabel("F_0 (Hz)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def export_pitch_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    FILENAME = "VOICEACTRESS100_002.wav"

    audio_file = BASE_DIR / "data" / "raw_audio" / "jsut_ver1.1" / "jsut_ver1.1" / "voiceactress100" / "wav" / FILENAME
    output_file = BASE_DIR / "data" / "processed" / "example_pitch.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_pitch = extract_pitch(audio_file)
    df_pitch = clean_pitch(df_pitch)
    df_pitch = normalize_time(df_pitch)

    df_resampled = resample_to_fixed_grid(df_pitch, n_points=100)
    pitch_vector = df_resampled["f0_resampled"].values
    print(pitch_vector.shape)

    raw_dir = BASE_DIR / "data" / "raw_audio" / "jsut_ver1.1" / "jsut_ver1.1" / "voiceactress100" / "wav"

    wav_files = sorted(raw_dir.glob("*.wav"))[:20]

    pitch_vectors = []

    for wav_path in wav_files:
        df_pitch = extract_pitch(wav_path)
        df_pitch = clean_pitch(df_pitch)
        df_pitch = normalize_time(df_pitch)
        df_resampled = resample_to_fixed_grid(df_pitch, n_points=100)

        pitch_vector = df_resampled["f0_resampled"].values
        pitch_vectors.append(pitch_vector)

    X = np.vstack(pitch_vectors)

    print("Shape of dataset:", X.shape)
    
    mean_contour = X.mean(axis=0)

    grid = np.linspace(0, 1, 100)

    plt.figure(figsize=(8,4))
    plt.plot(grid, mean_contour, linewidth=2)
    plt.title("Mean Pitch Contour (20 Utterances)")
    plt.xlabel("Normalized Time")
    plt.ylabel("F0 (Hz)")
    plt.tight_layout()
    plt.show()

    export_pitch_csv(df_pitch, output_file)

    print(f"Pitch extraction complete. Saved to {output_file}")
