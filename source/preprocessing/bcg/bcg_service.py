import numpy as np
import pandas as pd
import statistics

from source import utils
from source.constants import Constants
from source.preprocessing.motion.motion_collection import MotionCollection
from source.preprocessing.bcg.proell_functions import Proell


class BcgService(object):
    def get_cropped_file_path(subject_id):
        return Constants.CROPPED_FILE_PATH.joinpath(subject_id + "_cleaned_bcg.out")
    @staticmethod
    def build_hr_counts(subject_id, data):
        print("Getting BCG HR for subject " + str(subject_id) + "...")

        selected_data = data[:, :2] #timestamps and x columns only
        accel_df = pd.DataFrame(selected_data, columns=['timestamps', 'x'])

        dif = np.diff(np.asarray(accel_df['timestamps']))
        freqs = 1 / dif

        mode_value = statistics.mode(freqs)
        fs = round(mode_value * 2) / 2
        print(f"fs: {fs}")

        snippet_duration = 15  # 15 seconds
        step_size = 5  # 5 seconds

        #HR Stuff
        bcg_ts_hr = np.empty((0, 2))

        if fs >=49.5:
            for start_time in range(0, int(np.max(np.asarray(accel_df['timestamps']))), step_size):
                end_time = start_time + snippet_duration
                snippet_df = accel_df[(accel_df['timestamps'] >= start_time) & (accel_df['timestamps'] < end_time)]

                timestamps = np.asarray(snippet_df['timestamps'])

                if timestamps.any() and len(timestamps) > 0.9*fs*snippet_duration:
                    ts_dif = np.diff(timestamps)

                    if np.max(ts_dif) < 1:
                        accel_snip = np.asarray(snippet_df['x'])
                        mvmt_snip = Proell.detect_movements(accel_snip, fs)
                        accel_norm = (accel_snip - np.mean(accel_snip)) / np.std(accel_snip)
                        coarse_peaks, bcg_enhanced, bcg_coarse, ijk = Proell.segmenter(start_time, accel_norm, fs, show=False)
                        j_peaks = [array[1] for array in ijk]  # NOTE: not currently used. using coarse peaks instead
                        clean_j = Proell.reject_mvmt_peaks(coarse_peaks, mvmt_snip, mvmtDistance=10)
                        bcg_hr_temp, bcg_std_temp = Proell.heartrate_from_indices(clean_j, fs)

                        bcg_ts_hr = np.append(bcg_ts_hr, np.array([[start_time, bcg_hr_temp]]), axis=0)

            #save to text file
            bcg_output_path = BcgService.get_cropped_file_path(subject_id)
            np.savetxt(bcg_output_path, bcg_ts_hr, fmt='%f')

