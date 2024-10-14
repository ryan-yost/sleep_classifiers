import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import LeaveOneOut
import time
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


start_time = time.time()

# Define file paths and subject_ids
data_folder = "C:\\Users\\elemind1\\Documents\\GitHub\\sleep_classifiers\\outputs\\features_altini"
output_folder = "C:\\Users\\elemind1\\Documents\\GitHub\\sleep_classifiers\\outputs\\altini_trained_16Jan24"
subject_ids = ['3509524', '5132496', '5498603', '4018081', '9106476', '8686948', '4314139', '1818471', '8173033',
               '7749105', '759667', '8000685', '6220552', '844359', '1360686', '8692923']

def load_labels(subject_id):
    label_file = os.path.join(data_folder, f"{subject_id}_psg_labels.out")
    labels = np.loadtxt(label_file, dtype=float, delimiter=' ')

    # Collapse labels into 4 stages
    labels[labels == 1] = 1  # sleep
    labels[labels == 2] = 1  # sleep
    labels[labels == 3] = 1  # sleep
    labels[labels == 5] = 1  # sleep

    return labels

all_labels = []
all_features = []

print("Loading labels and features")

for subject_id in subject_ids:
    # Load labels
    labels = load_labels(subject_id)
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

# Initialize LGBMClassifier
lgbm = LGBMClassifier(boosting_type='dart', n_estimators=500)

metrics = {'SubId': [], 'Wake_Sensitivity': [], 'Wake_Specificity': [], 'Wake_TP': [], 'Wake_TN': [], 'Wake_FP': [], 'Wake_FN': [],}

# Leave-One-Out Cross-Validation
for test_subject_index, test_subject_id in enumerate(subject_ids):
    print(f"Leaving out subject {test_subject_id}")

    # Separate the test subject from the rest
    train_labels = all_labels[:test_subject_index] + all_labels[test_subject_index + 1:]
    train_labels = np.concatenate(train_labels)
    train_features = all_features[:test_subject_index] + all_features[test_subject_index + 1:]
    train_features = np.vstack(train_features)

    test_labels = all_labels[test_subject_index]
    test_features = all_features[test_subject_index]

    # oversample dataset to balance the datasets
    undersampler = RandomUnderSampler()
    train_features_resampled, train_labels_resampled = undersampler.fit_resample(train_features, train_labels)

    lgbm.fit(train_features_resampled, train_labels_resampled)

    y_pred = lgbm.predict(test_features)

    report = classification_report(test_labels, y_pred, labels=[0, 1], target_names=['Wake', 'Sleep'], output_dict=True)
    print(classification_report(test_labels, y_pred, labels=[0, 1], target_names=['Wake', 'Sleep']))

    conf_matrix = confusion_matrix(test_labels, y_pred)

    # Extract True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    TP_Wake = conf_matrix[0, 0]  # True Positives for "Wake"
    TN_Wake = conf_matrix[1, 1]  # True Negatives for "Wake"
    FP_Wake = conf_matrix[1, 0]  # False Positives for "Wake"
    FN_Wake = conf_matrix[0, 1]  # False Negatives for "Wake"

    # Note: sensitivty_wake = specificity_sleep, and vice versa
    sensitivity_Wake = TP_Wake / (TP_Wake + FN_Wake)
    specificity_Wake = TN_Wake / (TN_Wake + FP_Wake)

    print("Sensitivity for 'Wake':", sensitivity_Wake)
    print("Specificity for 'Wake':", specificity_Wake)

    metrics['SubId'].append(test_subject_id)
    metrics['Wake_Sensitivity'].append(sensitivity_Wake)
    metrics['Wake_Specificity'].append(specificity_Wake)
    metrics['Wake_TP'].append(TP_Wake)
    metrics['Wake_TN'].append(TN_Wake)
    metrics['Wake_FP'].append(FP_Wake)
    metrics['Wake_FN'].append(FN_Wake)

metrics_df = pd.DataFrame(metrics)

# Create a list of data to plot
wake_sensitivity = metrics_df['Wake_Sensitivity']
wake_specificity = metrics_df['Wake_Specificity']
data_to_plot = [wake_specificity, wake_sensitivity]

plt.figure(figsize=(8, 6))
boxprops = dict(color='blue')
plt.boxplot(data_to_plot, labels=['Sleep Sensitivity', 'Wake Sensitivity'], boxprops=boxprops)
plt.title('2-Class Sensitivity - LOOCV')
plt.ylabel('Values')
plt.show()

wake_TP = metrics_df['Wake_TP']
wake_TN = metrics_df['Wake_TN']
wake_FP = metrics_df['Wake_FP']
wake_FN = metrics_df['Wake_FN']

# Calculate the overall TP, TN, FP, and FN
overall_TP = wake_TP.sum()
overall_TN = wake_TN.sum()
overall_FP = wake_FP.sum()
overall_FN = wake_FN.sum()

# Create a confusion matrix as a DataFrame
confusion_matrix = pd.DataFrame({'Actual Sleep': [overall_TN, overall_FP], 'Actual Wake': [overall_FN, overall_TP]},
                                index=['Predicted Sleep', 'Predicted Wake'])

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest', aspect='auto')
plt.colorbar()
plt.title('2-Class Confusion Matrix - LOOCV')
plt.xlabel('Actual Class')
plt.ylabel('Predicted Class')
plt.xticks(np.arange(2), ['Sleep', 'Wake'])
plt.yticks(np.arange(2), ['Sleep', 'Wake'])

# Display the values in the heatmap
for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix.iloc[i, j], ha='center', va='center', color='black', fontsize=14)

plt.show()





