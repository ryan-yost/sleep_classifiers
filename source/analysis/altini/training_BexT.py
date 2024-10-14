import os
import numpy as np
import lightgbm as lgbm
import time

import pandas as pd

# Define file paths and subject_ids
data_folder = "C:\\Users\\elemind1\\Documents\\GitHub\\sleep_classifiers\\outputs\\features_altini"
output_folder = "C:\\Users\\elemind1\\Documents\\GitHub\\sleep_classifiers\\outputs\\altini_trained_16Jan24"
subject_ids_all = ['3509524', '9106476', '4018081', '1818471','5132496', '8686948', '4314139', '5498603', '8173033',
               '7749105', '759667', '8000685', '6220552', '844359', '1360686', '8692923']

def load_labels(subject_id):
    label_file = os.path.join(data_folder, f"{subject_id}_psg_labels.out")
    labels = np.loadtxt(label_file, dtype=float, delimiter=' ')
    # 4 stages -standard

    labels[labels == 2] = 1 # Shift N2
    labels[labels == 3] = 2  # Shift N3
    labels[labels == 5] = 3  # Shift REM down 1

    return labels


# Load all labels and features first
all_labels = []
all_features = []

print("Loading labels and features")

subject_ids = []
for subject_id  in subject_ids_all:
    # Load labels
    labels = load_labels(subject_id)
    if len(labels) > 600:
        subject_ids.append(subject_id)
        all_labels.append(labels)

        # Load features for this subject
        features_files = [
            f"{subject_id}_bcg_features.out",
            f"{subject_id}_accel_features.out",
            f"{subject_id}_cosine_feature.out",
            f"{subject_id}_decay_feature.out",
            f"{subject_id}_growth_feature.out"
        ]
        subject_features = []
        for file in features_files:
            feature_data = np.loadtxt(os.path.join(data_folder, file), dtype=float, delimiter=' ')
            if feature_data.ndim == 1:
                # Expand 1D feature to 2D
                feature_data = feature_data.reshape(-1, 1)
            subject_features.append(feature_data)

        subject_features = np.concatenate(subject_features, axis=1)
        all_features.append(subject_features)


y = np.concatenate(all_labels)
X = np.vstack(all_features)

X_enc = pd.get_dummies(X)

clf = lgbm.LGBMClassifier(objective="binary", n_estimators=500, random_state=1121218)

