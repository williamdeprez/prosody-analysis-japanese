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

def process_utterance(audio_path: Path, n_points: int = 100) -> np.ndarray:
    """
    Uses the full pipeline to extract a fixed-length pitch contour vector from an audio file. This includes pitch extraction, cleaning, time normalization, and resampling to a fixed grid.
    
    :param audio_path: File path to the input audio file
    :type audio_path: Path
    :param n_points: Number of points in the fixed-length pitch contour
    :type n_points: int
    :return: Pitch contour vector of length n_points
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """
    df = extract_pitch(audio_path)
    df = clean_pitch(df)
    df = normalize_time(df)
    df = resample_to_fixed_grid(df, n_points=n_points)

    return df["f0_resampled"].to_numpy(dtype=float)

def build_dataset(raw_dir: Path, n_files: int = 20, n_points: int = 100) -> np.ndarray:
    wav_files = sorted(raw_dir.glob("*.wav"))[:n_files]

    pitch_vectors = []

    for wav_path in wav_files:
        pitch_vector = process_utterance(wav_path, n_points)
        pitch_vectors.append(pitch_vector)

    return np.vstack(pitch_vectors)

def analyze_dataset(X: np.ndarray) -> None:
    grid = np.linspace(0, 1, X.shape[1])

    mean_contour = X.mean(axis=0)
    std_contour = X.std(axis=0)

    plt.figure(figsize=(8,4))
    plt.plot(grid, mean_contour, linewidth=2, label="Mean")

    plt.fill_between(
        grid,
        mean_contour - std_contour,
        mean_contour + std_contour,
        alpha=0.3,
        label="±1 Std Dev"
    )

    plt.title("Mean Pitch Contour")
    plt.xlabel("Normalized Time")
    plt.ylabel("F0 (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_variation_pca(X: np.ndarray, n_components: int = 2) -> None:
    grid = np.linspace(0, 1, X.shape[1])

    # Center dataset
    mean_contour = X.mean(axis=0)
    X_centered = X - mean_contour

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    plt.figure(figsize=(8,4))

    for i in range(n_components):
        plt.plot(grid, Vt[i], label=f"PC {i+1}")

    plt.title("Principal Modes of Pitch Variation")
    plt.xlabel("Normalized Time")
    plt.ylabel("Mode Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: print variance explained
    variance_explained = (S**2) / np.sum(S**2)
    print("Variance explained:")
    for i in range(n_components):
        print(f"PC {i+1}: {variance_explained[i]:.3f}")

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

    raw_dir = (
        BASE_DIR
        / "data"
        / "raw_audio"
        / "jsut_ver1.1"
        / "jsut_ver1.1"
        / "voiceactress100"
        / "wav"
    )

    # Build dataset of pitch vectors
    X = build_dataset(
        raw_dir=raw_dir,
        n_files=20,
        n_points=100
    )

    print("Dataset shape:", X.shape)

    # Analyze dataset (mean + variance band)
    analyze_dataset(X)
    analyze_variation_pca(X, n_components=2)