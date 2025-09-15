import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


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


def detect_note_onsets_from_mp4(mp4_file, sr=22050, backtrack=True, fmin=80, fmax=2000):
    # Charger l’audio depuis le mp4
    y, sr = librosa.load(mp4_file, sr=sr)

    # Détection d’enveloppe et d’onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=backtrack
    )

    hop = 512
    results = []

    for f in onset_frames:
        center = f * hop
        segment = y[center : center + int(0.1 * sr)]  # 100ms après l’onset
        if len(segment) < 100:
            continue

        # Estimer la fondamentale avec YIN
        f0 = librosa.yin(segment, fmin=fmin, fmax=fmax, sr=sr)
        f0 = f0[f0 > 0]  # garder les valeurs valides
        if len(f0) == 0:
            continue

        pitch_hz = np.median(f0)  # prendre la médiane pour plus de robustesse
        midi_note = int(librosa.hz_to_midi(pitch_hz))
        note_name = librosa.midi_to_note(midi_note)

        timestamp = librosa.frames_to_time(f, sr=sr)
        results.append(
            {"time": float(timestamp), "note": note_name, "freq": float(pitch_hz)}
        )

    return results


# onset_times = detect_onsets_from_mp4(
#    "videos_jonglage/vincent_6_balle_son_clean.mp4"
# )
# print("Catches détectés :", onset_times[])


# results = detect_note_onsets_from_mp4(
#    "videos_jonglage/vincent_3_couleur_avec_gants_clean.mp4"
# )
#
# for r in results:
#    print(f"time {r['time']:.2f}s → {r['note']} ({r['freq']:.1f} Hz)")


# results = detect_note_onsets_from_mp4(
#    "videos_jonglage/vincent_3_couleur_avec_gants_clean.mp4"
# )
#
# formatted = [
#    {"time": round(r["time"], 2), "freq": round(r["freq"], 1)} for r in results
# ]
#
# print(formatted)
