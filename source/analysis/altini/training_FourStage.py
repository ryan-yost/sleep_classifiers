import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgbm
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
subject_ids_all = ['3509524', '9106476', '4018081', '1818471','5132496', '8686948', '4314139', '5498603', '8173033',
               '7749105', '759667', '8000685', '6220552', '844359', '1360686', '8692923']


# Function to load labels, collapse stages, and one-hot encode
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

recall_metrics = {'SubId': [], 'N1/N2': [], 'N3': [], 'Wake': [], 'REM': []}
specitivity_metrics = {'SubId': [], 'N1/N2': [], 'N3': [], 'Wake': [], 'REM': []}

# Initialize LGBMClassifier
class_weights = {0: 1, 1: 0.25, 2: 5, 3: 5} #default:1
#lgbm = LGBMClassifier(boosting_type='dart', n_estimators=500, class_weight=class_weights)
lgbm_clf = lgbm.LGBMClassifier(n_estimators=1000, class_weight=class_weights)

for test_subject_index, test_subject_id in enumerate(subject_ids):
    print(f"Leaving out subject {test_subject_id}")

    # Separate the test subject from the rest
    train_labels = all_labels[:test_subject_index] + all_labels[test_subject_index + 1:]
    train_features = all_features[:test_subject_index] + all_features[test_subject_index + 1:]

    test_labels = all_labels[test_subject_index]
    test_features = all_features[test_subject_index]

    # Concatenate the lists for training
    train_labels = np.concatenate(train_labels)
    train_features = np.vstack(train_features)

    # oversaample dataset to balance the datasets
    sampler = RandomOverSampler()
    train_features_resampled, train_labels_resampled = sampler.fit_resample(train_features, train_labels)

    # Reshape train_labels to a 1D array
    eval_set = [(test_features, test_labels)]
    lgbm_clf.fit(train_features_resampled, train_labels_resampled, eval_set=eval_set)

    y_pred = lgbm_clf.predict(test_features)
    labels = [0, 1, 2, 3]
    target_names = ['Wake', 'N1/N2', 'N3', 'REM']
    report = classification_report(test_labels, y_pred, labels=labels, target_names=target_names,
                                   output_dict=True)

    # Calculate Cohen's Kappa for this subject
    subject_cohen_kappa = cohen_kappa_score(test_labels, y_pred)
    print(subject_cohen_kappa)

    # Print individual classification report
    print(f"Classification Report for Test Subject {test_subject_id}:")
    print(report)

    recall_metrics['SubId'].append(test_subject_id)
    recall_metrics['Wake'].append(report['Wake']['recall'])
    recall_metrics['N1/N2'].append(report['N1/N2']['recall'])
    recall_metrics['N3'].append(report['N3']['recall'])
    recall_metrics['REM'].append(report['REM']['recall'])

    specitivity_metrics['SubId'].append(test_subject_id)
    for label, class_name in zip(labels, target_names):
        # Create a confusion matrix for the current class
        class_labels = (test_labels == label)
        class_predictions = (y_pred == label)
        tn = np.sum(np.logical_and(~class_labels, ~class_predictions))
        fp = np.sum(np.logical_and(~class_labels, class_predictions))

        # Calculate specificity for the current class
        specificity = tn / (tn + fp)
        specitivity_metrics[class_name].append(specificity)

# Create a DataFrame from the recall metrics
recall_df = pd.DataFrame(recall_metrics)

# Box and whisker plots for recall
plt.figure(figsize=(4, 6))
boxprops = dict(color='blue')
recall_df.boxplot(column=['N1/N2', 'N3', 'Wake', 'REM'], boxprops=boxprops)
plt.title('4-class Sensitivity - LOOCV')
plt.ylabel('Sensitivity')
plt.xlabel('Class')
plt.xticks([1, 2, 3, 4], ['N1/N2', 'N3', 'Wake', 'REM'])
plt.show()

# Save the recall metrics dataframe as a CSV file
recall_df.to_csv(os.path.join(output_folder, 'recall_metrics.csv'), index=False)

specificity_df = pd.DataFrame(specitivity_metrics)

# Box and whisker plots for recall
plt.figure(figsize=(4, 6))
boxprops = dict(color='blue')
specificity_df.boxplot(column=['N1/N2', 'N3', 'Wake', 'REM'], boxprops=boxprops)
plt.title('4-class Specificity - LOOCV')
plt.ylabel('Specificity')
plt.xlabel('Class')
plt.xticks([1, 2, 3, 4], ['N1/N2', 'N3', 'Wake', 'REM'])
plt.show()