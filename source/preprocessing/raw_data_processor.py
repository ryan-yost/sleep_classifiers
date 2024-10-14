import numpy as np

from source import utils
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService
from source.preprocessing.bcg.bcg_service import BcgService
from source.preprocessing.interval import Interval
from source.preprocessing.motion.motion_service import MotionService
from source.preprocessing.psg.psg_service import PSGService
from source.sleep_stage import SleepStage



class RawDataProcessor:
    BASE_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')

    @staticmethod
    def crop_all(subject_id):
        # psg_raw_collection = PSGService.read_raw(subject_id)       # Used to extract PSG details from the reports
        psg_raw_collection = PSGService.read_precleaned(subject_id)  # Loads already extracted PSG data
        motion_collection = MotionService.load_raw(subject_id)
        heart_rate_collection = HeartRateService.load_raw(subject_id)

        valid_interval = RawDataProcessor.get_intersecting_interval([psg_raw_collection,
                                                                     motion_collection,
                                                                     heart_rate_collection])

        psg_raw_collection = PSGService.crop(psg_raw_collection, valid_interval)
        motion_collection = MotionService.crop(motion_collection, valid_interval)
        heart_rate_collection = HeartRateService.crop(heart_rate_collection, valid_interval)

        PSGService.write(psg_raw_collection)
        MotionService.write(motion_collection)
        HeartRateService.write(heart_rate_collection)
        ActivityCountService.build_activity_counts_without_matlab(subject_id, motion_collection.data)  # Builds activity counts with python, not MATLAB

        # RY New: given cropped actigraphy files, get HR
        BcgService.build_hr_counts(subject_id, motion_collection.data)

    @staticmethod
    def get_intersecting_interval(collection_list):
        start_times = []
        end_times = []
        for collection in collection_list:
            interval = collection.get_interval()
            start_times.append(interval.start_time)
            end_times.append(interval.end_time)

        return Interval(start_time=max(start_times), end_time=min(end_times))

    @staticmethod
    def get_valid_epochs(subject_id):

        psg_collection = PSGService.load_cropped(subject_id)
        motion_collection = MotionService.load_cropped(subject_id)
        heart_rate_collection = HeartRateService.load_cropped(subject_id)

        start_time = psg_collection.data[0].epoch.timestamp
        motion_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(motion_collection.timestamps,
                                                                              start_time,50)
        hr_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(heart_rate_collection.timestamps,
                                                                          start_time,0.2)

        valid_epochs = []
        for stage_item in psg_collection.data:
            epoch = stage_item.epoch

            if epoch.timestamp in motion_epoch_dictionary and epoch.timestamp in hr_epoch_dictionary \
                    and stage_item.stage != SleepStage.unscored:
                valid_epochs.append(epoch)

        return valid_epochs

    @staticmethod
    def get_valid_epoch_dictionary(timestamps, start_time, fs):
        empty_ts = np.arange(start_time, timestamps[-1], 30)

        epoch_dictionary = {}

        for empty_start in empty_ts:
            #print('\n')
            start_idx = np.searchsorted(timestamps, empty_start)
            #print('empty_start: ', empty_start)
            # print('start_idx: ', start_idx)
            #print('start_ts: ', timestamps[start_idx])

            empty_end = empty_start + 30.0
            end_idx = np.searchsorted(timestamps, empty_end, side='right') - 1

            # print('empty_end: ', empty_end)
            # print('end_idx: ', end_idx)
            #print('end_ts: ', timestamps[end_idx])

            if start_idx is not None and end_idx is not None:
                snip = timestamps[start_idx:end_idx + 1]
                #print('len(snip): ', len(snip))
                if len(snip) >= np.floor(fs * 30 * 0.95):
                    epoch_dictionary[empty_start] = True
                    #print('TRUE')
        return epoch_dictionary