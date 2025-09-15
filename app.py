import os
import sys
import streamlit as st

# Supprime les logs verbeux de TF (gardera warnings/erreurs seulement)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------
# Gestion du state
# -------------------
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "init_triggered" not in st.session_state:
    st.session_state.init_triggered = False
if "main_func" not in st.session_state:
    st.session_state.main_func = None

VIDEO_DIR = "videos"

# -------------------
# Sidebar : paramètres
# -------------------
st.sidebar.header("⚙️ Paramètres de la pipeline")

model = st.sidebar.text_input(
    "Chemin du modèle (--model)", "grid_models/grid_model_submovavg_64x64.h5"
)
n_balls = st.sidebar.number_input("Nombre de balles (--n-balls)", min_value=1, value=3, step=1)
fps = st.sidebar.number_input("FPS (--fps)", min_value=1, value=30)
max_s = st.sidebar.number_input("Hauteur max siteswap (--max-s)", min_value=1, value=12)
sigma = st.sidebar.number_input("Sigma (--sigma)", min_value=0.0, value=0.25, step=0.05)
win = st.sidebar.number_input("Fenêtre tempo local (--win)", min_value=1, value=7, step=2)
beam_width = st.sidebar.number_input("Largeur beam-search (--beam-width)", min_value=1, value=200)
W = st.sidebar.number_input("Taille fenêtre d’occupation (--W)", min_value=1, value=24)
print_json = st.sidebar.checkbox("Afficher JSON (--print-json)")
plot_tempo = st.sidebar.checkbox("Afficher la courbe de tempo (--plot-tempo)")
save_tempo = st.sidebar.text_input(
    "Chemin pour sauvegarder la courbe (--save-tempo)", "plot.png"
)

# -------------------
# Upload vidéo
# -------------------
uploaded_video = st.file_uploader("Choisis une vidéo", type=["mp4", "avi", "mov"])

video_path = None
if uploaded_video is not None:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    video_path = os.path.join(VIDEO_DIR, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.video(video_path)

# -------------------
# Initialisation pipeline (non bloquante)
# -------------------
if not st.session_state.pipeline_ready:
    if not st.session_state.init_triggered:
        st.session_state.init_triggered = True
        st.info("⏳ Préparation de la pipeline en arrière-plan...")
        st.rerun()
    else:
        st.info("⚙️ Initialisation de TensorFlow et de la pipeline...")
        sys.path.append(os.getcwd())
        from pipeline.main_pipeline import main
        st.session_state.main_func = main
        st.session_state.pipeline_ready = True
        st.success("✅ Pipeline prête 🎉")

# -------------------
# Bouton lancer
# -------------------
launch_btn = st.button("▶️ Lancer la pipeline", disabled=not st.session_state.pipeline_ready)

if launch_btn:
    if video_path is None:
        st.warning("⚠️ Merci d'importer une vidéo avant de lancer la pipeline.")
    else:
        with st.spinner("Analyse en cours..."):
            argv = [
                "main_pipeline.py",
                "--video", video_path,
                "--model", model,
                "--n-balls", str(n_balls),
                "--max-s", str(max_s),
                "--sigma", str(sigma),
                "--win", str(win),
                "--beam-width", str(beam_width),
                "--W", str(W),
            ]
            if fps > 0:
                argv += ["--fps", str(fps)]
            if print_json:
                argv.append("--print-json")
            if plot_tempo:
                argv.append("--plot-tempo")
            if save_tempo:
                argv += ["--save-tempo", save_tempo]

            old_argv = sys.argv
            sys.argv = argv
            try:
                # --- Appel unique de main() ---
                results = st.session_state.main_func()
            except Exception as e:
                st.error(f"Erreur dans la pipeline : {e}")
                results = None
            finally:
                sys.argv = old_argv

        # --- Affichage des résultats ---
        if results:
            st.subheader("📋 Événements détectés")
            st.json(results["events"])

            st.subheader("🎲 Résultat Viterbi")
            st.json(results["viterbi_result"])

            st.subheader("✅ Validation")
            st.write(results["validation"])

            if len(results["tempo"]["t_mid"]) > 0:
                st.subheader("📈 Courbe de tempo")
                st.line_chart({"BPM": results["tempo"]["bpm_local"]})