import librosa
import numpy as np
from sklearn.cluster import KMeans


def detect_spectral_onsets_from_mp4(
    mp4_file, sr=22050, backtrack=True, n_clusters=None
):
    """
    Detects onsets in a video file and represents them by their spectral fingerprint.
    Optionally clusters them to identify which onset corresponds to which ball.
    """
    # Load audio
    y, sr = librosa.load(mp4_file, sr=sr)

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=backtrack
    )

    hop = 512
    results = []
    spectra = []

    for f in onset_frames:
        center = f * hop
        segment = y[center : center + int(0.1 * sr)]  # 100 ms segment
        if len(segment) < 100:
            continue

        # Spectrum (power spectrum)
        S = np.abs(librosa.stft(segment, n_fft=2048)) ** 2
        spec = np.mean(S, axis=1)  # average across time
        spec_norm = spec / (np.linalg.norm(spec) + 1e-8)

        spectra.append(spec_norm)

        timestamp = librosa.frames_to_time(f, sr=sr)
        results.append({"time": float(timestamp), "spectrum": spec_norm})

    # Optional clustering (group by ball timbre)
    if n_clusters is not None and len(spectra) >= n_clusters:
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(spectra)
        labels = km.labels_
        for i, r in enumerate(results):
            r["cluster"] = int(labels[i])
    else:
        for r in results:
            r["cluster"] = None

    return results


# results = detect_spectral_onsets_from_mp4(
#    "videos_jonglage/vincent_6_balle_son_clean.mp4",
#    n_clusters=6,  # if you expect 3 distinct balls/notes
# )
#
# for r in results:
#    print(f"time {r['time']:.2f}s → cluster {r['cluster']}")
