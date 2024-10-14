from source import utils
from source.preprocessing.motion.motion_service import MotionService
from source.preprocessing.psg.psg_service import PSGService
import numpy as np
import statistics
from source.sleep_stage import SleepStage


class RawDataProcessor:
    BASE_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')

    def get_valid_epochs(subject_id):

        psg_collection = PSGService.load_cropped(subject_id)
        motion_collection = MotionService.load_cropped(subject_id)

        start_time = psg_collection.data[0].epoch.timestamp

        motion_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(motion_collection.timestamps,
                                                                              start_time)
        valid_epochs = []
        for stage_item in psg_collection.data:
            epoch = stage_item.epoch

            if epoch.timestamp in motion_epoch_dictionary and stage_item.stage != SleepStage.unscored:
                if subject_id == '6220552' and (epoch.timestamp == 13650 or epoch.timestamp >= 14790):
                    # TODO: this removes ~300 epochs. consider a more elegant solution
                    print('troubling epoch removed')
                elif subject_id == '8692923' and (epoch.timestamp == 1110 or 1800 <= epoch.timestamp <= 2400 or 5280 <= epoch.timestamp <= 5880 or 9810 <= epoch.timestamp <= 10410 or 17910 <= epoch.timestamp <= 18510 or 18900 <= epoch.timestamp <= 18900 + 20*30 or epoch.timestamp >= 20730):
                    # TODO: this too
                    print('troubling epoch removed')
                else:
                    valid_epochs.append(epoch)

        return valid_epochs

    @staticmethod
    def get_valid_epoch_dictionary(timestamps, start_time):

        empty_ts = np.arange(start_time, timestamps[-1], 30)

        epoch_dictionary = {}

        for empty_start in empty_ts:
            start_idx = np.searchsorted(timestamps, empty_start - 135)
            empty_end = empty_start + 30.0 + 135
            end_idx = np.searchsorted(timestamps, empty_end, side='right') - 1

            # must check 30s epoch too
            start_micro = np.searchsorted(timestamps, empty_start)
            end_micro = np.searchsorted(timestamps, empty_start + 30.0, side='right') - 1

            if start_idx is not None and end_idx is not None:
                snip = timestamps[start_idx:end_idx + 1]
                micro_snip = timestamps[start_micro:end_micro + 1]
                if len(snip) > 10:
                    dif_array = np.diff(np.asarray(snip))
                    freq_array = 1 / dif_array
                    accel_fs = round(statistics.mode(freq_array) * 2) / 2

                    if 30.0 < accel_fs < 75.0 and len(snip) >= np.floor(accel_fs * 5 * 60 * 0.95) and max(dif_array) < 0.15 and len(micro_snip) >= np.floor(accel_fs * 30 * 0.8):
                        epoch_dictionary[empty_start] = True

        return epoch_dictionary