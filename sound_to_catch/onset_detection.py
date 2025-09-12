import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as npp


def detect_onsets_from_mp4(mp4_file, sr=22050, backtrack=True, plot=True):
    # Charger directement l'audio depuis le mp4
    y, sr = librosa.load(mp4_file, sr=sr)

    # Calculer l'enveloppe d'énergie d'onset
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Détecter les onsets
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=backtrack
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    if plot:
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.vlines(
            onset_times, ymin=-1, ymax=1, color="r", linestyle="--", label="Catches"
        )
        plt.title("Détection des catches (onsets audio)")
        plt.legend()
        plt.show()

    return onset_times


onset_times = detect_onsets_from_mp4(
    "videos_jonglage/vincent_6_balle_son_clean.mp4", plot=False
)
print("Catches détectés :", onset_times)
