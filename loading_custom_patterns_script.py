import pickle
import numpy as np
import pandas as pd
import os
import tfmindi

# TF-MInDi import
from tfmindi import load_motif_collection

# -------------------------------
# 1. Load per-group PWMs
# -------------------------------
with open("./../per_group_ppms.pkl", "rb") as f:
    group_pwms = pickle.load(f)

flat_collection = {}
records = []

for group, motifs in group_pwms.items():
    for motif_name, pwm in motifs.items():
        motif_id = f"{group}_{motif_name}"
        
        # Convert to numpy array
        pwm = np.array(pwm)
        
        # Ensure PWM shape is (4, L)
        if pwm.ndim == 2 and pwm.shape[1] == 4 and pwm.shape[0] != 4:
            pwm = pwm.T
        
        flat_collection[motif_id] = pwm
        
        records.append({
            "MotifID": motif_id,
            "Group": group,
            "OriginalMotif": motif_name
        })

# -------------------------------
# 2. Create annotation dataframe
# -------------------------------
motif_annotations = pd.DataFrame(records).set_index("MotifID")

# -------------------------------
# 3. Print summary and first few motifs
# -------------------------------
print("Number of motifs loaded:", len(flat_collection))
print("\nFirst few motif annotations:")
print(motif_annotations.head())

print("\nFirst few motif PWMs:")
for i, (motif_id, pwm) in enumerate(flat_collection.items()):
    if i >= 5:  # print only first 5
        break
    print(f"\nMotif ID: {motif_id}")
    print(pwm)

# Comparison to motif load function ##########################


# -------------------------------
# 4. Load sampled TF-MInDi motifs
# -------------------------------
import tfmindi as tm

# Fetch default TF-MInDi collection + annotations
motif_collection_dir = tm.fetch_motif_collection()
motif_annotations_file = tm.fetch_motif_annotations()

# Load sampled motif names
motif_samples_path = "sampled_motifs.txt"
with open(motif_samples_path) as f:
    motif_names = [line.strip() for line in f.readlines()]

print("\nNumber of sampled motif names:", len(motif_names))
print("\nFirst 5 motif names from sampled_motifs.txt:")
print(motif_names[:5])

# Load only sampled motifs
motif_collection = tm.load_motif_collection(
    motif_collection_dir,
    motif_names=motif_names
)

motif_annotations = tm.load_motif_annotations(motif_annotations_file)
#motif_to_db = tm.load_motif_to_dbd(motif_annotations)

print("\nNumber of sampled motifs loaded from TF-MInDi:", len(motif_collection))

print("\nFirst 5 sampled motifs (ID + PWM shape):")
for i, (motif_id, pwm) in enumerate(motif_collection.items()):
    if i >= 5:
        break
    print(f"\nMotif ID: {motif_id}")
    print("PWM shape:", np.array(pwm).shape)
    print(np.array(pwm))
