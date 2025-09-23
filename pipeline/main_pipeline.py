import argparse
import json
from typing import List, Dict, Any

from cnn_detections.main_analyzer import find_catch_events
from catch_to_siteswap.inference_model_viterbi import (
    viterbi_siteswap_beam,
    validate_siteswap_from_assignments,
)
from catch_to_siteswap.tempo_curve import tempo_curve

import matplotlib.pyplot as plt


def build_viterbi_events(
    catch_events: List[Dict[str, Any]], n_balls: int, fps: float | None
) -> List[Dict[str, Any]]:
    """
    Convert catch_events -> events list required by viterbi_siteswap_beam:
      [{"time": float, "ball_id": int}, ...] sorted by time.

    - time priority: ev["catch_time"] if present, else ev["catch_frame"]/fps (requires fps)
    - ball_id: use ev["ball_id"] if present; otherwise assign cyclicly 1..n_balls as fallback.
    """
    events = []

    # detect if ball_id is present at least once
    has_ball_id = any(
        "ball_id" in ev and ev["ball_id"] is not None for ev in catch_events
    )

    for idx, ev in enumerate(catch_events):
        item = {}

        if "catch_time" in ev and ev["catch_time"] is not None:
            item["time"] = float(ev["catch_time"])
        elif "catch_frame" in ev and ev["catch_frame"] is not None and fps:
            item["time"] = float(ev["catch_frame"]) / float(fps)
        else:
            raise ValueError(
                "Impossible de construire 'time' pour un événement: besoin de catch_time "
                "ou de catch_frame + --fps."
            )

        if has_ball_id and ("ball_id" in ev) and (ev["ball_id"] is not None):
            item["ball_id"] = int(ev["ball_id"])
        else:
            # Fallback: assign cyclic IDs 1..n_balls
            item["ball_id"] = (idx % n_balls) + 1

        events.append(item)

    # sort by time (required)
    events.sort(key=lambda e: e["time"])
    return events


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: detection -> Viterbi inference (+ tempo curve)"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Chemin de la vidéo d'entrée"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="grid_models/grid_model_submovavg_64x64.h5",
        help="Chemin du modèle de grille",
    )
    parser.add_argument("--n-balls", type=int, required=True, help="Nombre de balles")
    parser.add_argument(
        "--fps", type=float, default=None, help="FPS (si pas de catch_time)"
    )
    parser.add_argument("--max-s", type=int, default=12, help="Hauteur max de siteswap")
    parser.add_argument(
        "--sigma", type=float, default=0.25, help="Ecart-type observation (en beats)"
    )
    parser.add_argument(
        "--win", type=int, default=7, help="Fenêtre du tempo local (odd recommandé)"
    )
    parser.add_argument(
        "--beam-width", type=int, default=200, help="Largeur du beam-search"
    )
    parser.add_argument(
        "--W", type=int, default=24, help="Taille de la fenêtre d’occupation"
    )
    parser.add_argument(
        "--print-json", action="store_true", help="Afficher le JSON des événements"
    )
    parser.add_argument(
        "--plot-tempo", action="store_true", help="Afficher la courbe de tempo"
    )
    parser.add_argument(
        "--save-tempo",
        type=str,
        default=None,
        help="Chemin pour sauvegarder la courbe (PNG)",
    )
    args = parser.parse_args()

    # 1) Détection des catches (frames/temps/ball_id si dispo)
    catch_events = find_catch_events(
        video_path=args.video,
        model_path=args.model,
        n_balls=args.n_balls,
    )

    # 2) Construire les événements pour Viterbi (time + ball_id)
    events = build_viterbi_events(catch_events, n_balls=args.n_balls, fps=args.fps)

    if args.print_json:
        print(
            json.dumps(
                {"n_balls": args.n_balls, "events": events},
                indent=2,
                ensure_ascii=False,
            )
        )

    # 3) Inférence Viterbi (même structure de résultat que inference_model_viterbi)
    res = viterbi_siteswap_beam(
        events=events,
        max_s=args.max_s,
        sigma=args.sigma,
        win=args.win,
        beam_width=args.beam_width,
        W=args.W,
    )
    print("Méthode:", res["method"])
    print("Assignments (par lancer):", res["assignments"])
    print("Residuals:", [round(x, 3) for x in res["residuals"]])
    print("Score:", res["score"])

    # Validation simple (comme dans inference_model_viterbi)
    val = validate_siteswap_from_assignments(res["assignments"])
    print("Validation:", val)

    # 4) Courbe de tempo (via tempo_curve)
    t_mid, bpm_local = tempo_curve(events, win=args.win, method="median")
    if len(t_mid) == 0:
        print("Pas assez d'événements pour estimer le tempo.")
    else:
        if args.plot_tempo or args.save_tempo:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t_mid, bpm_local, marker="o", linestyle="-", label="Tempo local")
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Tempo (BPM)")
            ax.set_title("Courbe de tempo de la prestation")
            ax.legend()
            if args.save_tempo:
                fig.savefig(args.save_tempo, dpi=150, bbox_inches="tight")
                print(f"Courbe de tempo sauvegardée: {args.save_tempo}")
            if args.plot_tempo:
                plt.show()
            else:
                plt.close(fig)
    
    return {
        "events": events,
        "viterbi_result": res,
        "validation": val,
        "tempo": {"t_mid": t_mid, "bpm_local": bpm_local},
    }


if __name__ == "__main__":
    main()
