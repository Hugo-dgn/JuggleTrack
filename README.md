# Installation

## Installation (avec Poetry)

### 1. Prérequis (macOS)
- macOS 
- Python 3.12 installé (si nécessaire : `brew install python@3.12`)
- FFmpeg (pour l’audio/vidéo) :
```bash
brew install ffmpeg
```

### 2. Installer Poetry (si pas déjà installé)
```bash
pip install poetry
```

ou 

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Ajoute Poetry au PATH (si nécessaire, fichier `~/.zshrc`) :
```bash
export PATH="$HOME/.local/bin:$PATH"
```
Recharge :
```bash
source ~/.zshrc
poetry --version
```

### 3. Cloner le dépôt
```bash
git clone <URL_DU_DEPOT>
cd JuggleTrack
```

### 4. Installation des dépendances
Installation normale :
```bash
poetry install
```


### 5. Gestion de TensorFlow sur macOS (Apple Silicon)
Les packages `tensorflow` standards peuvent poser problème. Utilise la version macOS :
```bash
poetry remove tensorflow
poetry add tensorflow-macos
```
(Optionnel Metal GPU) :
```bash
poetry add tensorflow-metal
```

Si un message “lock file is currently blocking” / “lock blocking” ou conflit de version apparaît :
```bash
rm poetry.lock
poetry lock
poetry install
```

### 6. Activer l’environnement
Deux méthodes :
```bash
poetry env activate          # ouvre un shell virtuel ou donne la commande à lancer pour l'ouvrir
# ou sans activer directement :
poetry run python -V
```


### 7. Lancer l’application Streamlit
```bash
poetry run streamlit run app.py
```

### 8. Résolution de problèmes fréquents
| Problème | Solution |
|----------|----------|
| ModuleNotFoundError | Toujours lancer avec `poetry run ...` |
| TensorFlow KO | Utiliser `tensorflow-macos` + éventuellement `tensorflow-metal` |
| ffmpeg introuvable | `brew install ffmpeg` |

### 9. Mise à jour des dépendances
```bash
poetry update
```

### 10. Exécuter un script arbitraire
```bash
poetry run python path/to/script.py
```

### 13. Quitter l’environnement
```bash
deactivate    # si dans 'poetry shell'
```
