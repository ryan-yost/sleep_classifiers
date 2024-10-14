import time
from source.preprocessing.feature_builder_altini import FeatureBuilder

def run_preprocessing(subject_set):
    start_time = time.time()

    for subject in subject_set:
        FeatureBuilder.build(subject)

    end_time = time.time()
    print("Execution took " + str((end_time - start_time) / 60) + " minutes")

# subject_ids = ['3509524', '5132496', '5498603', '4018081', '9106476', '8686948', '4314139', '1818471', '8173033',
#               '7749105', '759667', '8000685', '6220552', '844359', '1360686', '8692923']

subject_ids = ['8692923']

run_preprocessing(subject_ids)


