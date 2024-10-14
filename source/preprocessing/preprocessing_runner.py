import time

from source.analysis.figures.data_plot_builder import DataPlotBuilder
from source.analysis.setup.subject_builder import SubjectBuilder
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.feature_builder import FeatureBuilder
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.circadian_service import CircadianService


def run_preprocessing(subject_set):
    start_time = time.time()

    for subject in subject_set:
         print("Cropping data from subject " + str(subject) + "...")
         RawDataProcessor.crop_all(str(subject))

    #if Constants.INCLUDE_CIRCADIAN:
    #    ActivityCountService.build_activity_counts()  # This uses MATLAB, but has been replaced with a python implementation
    #    CircadianService.build_circadian_model()      # Both of the circadian lines require MATLAB to run
    #    CircadianService.build_circadian_mesa()       # INCLUDE_CIRCADIAN = False by default because most people don't have MATLAB

    for subject in subject_set:
        FeatureBuilder.build(str(subject))

    end_time = time.time()
    print("Execution took " + str((end_time - start_time) / 60) + " minutes")


#subject_ids = SubjectBuilder.get_all_subject_ids()
subject_ids = ['3509524', '5132496', '5498603', '4018081', '9106476', '8686948', '4314139', '1818471', '8173033',
              '7749105', '759667', '8000685', '6220552', '844359', '1360686', '8692923']
run_preprocessing(subject_ids)

#for subject_id in subject_ids:
#    DataPlotBuilder.make_data_demo(subject_id, False)
