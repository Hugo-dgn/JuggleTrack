import pandas as pd

# Path to your CSV
file_path = "/home/hugo/Documents/JuggleTrack/dataset/labels/ss_64x_id881.csv"

# Load CSV
df = pd.read_csv(file_path)

# Inspect the first few rows
print(df.head())
