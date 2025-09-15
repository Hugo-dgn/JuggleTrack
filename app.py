import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

        # --- Vérifier que les résultats existent ---
        if results:

            # -------------------
            # 1️⃣ Événements détectés
            # -------------------
            events_df = pd.DataFrame(results["events"])
            st.subheader("📋 Événements détectés")
            st.dataframe(events_df)  # tableau scrollable et triable

            # -------------------
            # 2️⃣ Résultat Viterbi
            # -------------------
            viterbi = results["viterbi_result"]

            # Assignments
            st.subheader("🎲 Assignments par lancer")
            assignments_df = pd.DataFrame({
                "Event": range(len(viterbi["assignments"])),
                "Siteswap": viterbi["assignments"]
            })
            st.bar_chart(assignments_df.set_index("Event"))

            # Residuals
            st.subheader("📊 Residuals par lancer")
            residuals_df = pd.DataFrame({
                "Event": range(len(viterbi["residuals"])),
                "Residual": viterbi["residuals"]
            })
            st.line_chart(residuals_df.set_index("Event"))

            # Throws
            throws_df = pd.DataFrame(viterbi["throws"])
            st.subheader("🎯 Détails des lancers")
            st.dataframe(throws_df)

            # -------------------
            # 3️⃣ Validation
            # -------------------
            val = results["validation"]
            st.subheader("✅ Validation")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Longueur", val["length"])
            col2.metric("Moyenne balles", round(val["mean_balls"], 2))
            col3.metric("Moyenne OK", val["mean_ok"])
            col4.metric("Validité", "✅" if val["valid"] else "❌")

            if val.get("collisions", False):
                st.warning("⚠️ Des collisions ont été détectées")

            # -------------------
            # 4️⃣ Courbe de tempo
            # -------------------
            t_mid = results["tempo"]["t_mid"]
            bpm_local = results["tempo"]["bpm_local"]

            if t_mid is not None and len(t_mid) > 0:
                st.subheader("📈 Courbe de tempo")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(t_mid, bpm_local, marker="o", linestyle="-", color="orange", label="Tempo local")
                ax.set_xlabel("Temps (s)")
                ax.set_ylabel("BPM")
                ax.set_title("Courbe de tempo de la prestation")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)