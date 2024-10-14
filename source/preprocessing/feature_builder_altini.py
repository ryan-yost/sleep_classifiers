import math

import numpy as np
import scipy
from scipy.signal import butter, lfilter

from source.preprocessing.raw_data_processor_altini import RawDataProcessor
from source.preprocessing.psg.psg_label_service import PSGLabelService
from source.constants import Constants
from source import utils
from source.preprocessing.motion.motion_service import MotionService
import pandas as pd
import statistics
from source.preprocessing.bcg.proell_functions import Proell
from scipy.signal import welch, find_peaks
from scipy.signal import convolve
from scipy.fft import fft

import matplotlib.pyplot as plt



class FeatureBuilder(object):
    CROPPED_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')
    @staticmethod
    def build(subject_id):
        #print(f"Getting valid epochs for sub {subject_id}")
        valid_epochs = RawDataProcessor.get_valid_epochs(subject_id)
        print(len(valid_epochs))

        print(f"Building features for sub {subject_id}")
        FeatureBuilder.build_labels(subject_id, valid_epochs)
        FeatureBuilder.build_from_time(subject_id, valid_epochs)
        FeatureBuilder.build_from_accel(subject_id, valid_epochs)
        return

    def build_labels(subject_id, valid_epochs):
        psg_labels = PSGLabelService.build(subject_id, valid_epochs)
        FeatureBuilder.altini_psgwriter(subject_id, psg_labels)
        return

    def altini_psgwriter(subject_id, psg_labels):
        psg_labels_path = Constants.ALTINI_FEATURE_FILE_PATH.joinpath(subject_id + '_psg_labels.out')
        np.savetxt(psg_labels_path, psg_labels, fmt='%f')

    def build_from_time(subject_id, valid_epochs):
        cosine_feature = FeatureBuilder.build_circadian_feature(valid_epochs, 'cosine')
        decay_feature = FeatureBuilder.build_circadian_feature(valid_epochs, 'decay')
        growth_feature = FeatureBuilder.build_circadian_feature(valid_epochs, 'growth')

        FeatureBuilder.write_circadian_features(subject_id, cosine_feature, decay_feature, growth_feature)

        return

    def build_circadian_feature(valid_epochs, curve_type):
        features = []
        first_timestamp = valid_epochs[0].timestamp

        if curve_type == 'cosine':
            curve_function = FeatureBuilder.cosine_curve
        elif curve_type == 'decay':
            curve_function = FeatureBuilder.decay_curve
        elif curve_type == 'growth':
            curve_function = FeatureBuilder.growth_curve
        else:
            raise ValueError("Invalid curve_type. Supported types are 'cosine', 'decay', and 'growth'.")

        for epoch in valid_epochs:
            value = curve_function(epoch.timestamp - first_timestamp)
            normalized_value = value
            features.append(normalized_value)

        return np.array(features)

    def cosine_curve(time):
        # cosine with period of 24 hours
        sleep_drive_cosine_shift = 5
        return np.cos((time - sleep_drive_cosine_shift * 3600) * 2 * np.pi / (3600*24))

    def decay_curve(time):
        decay_rate = -np.log(0.05) / (8.333 * 3600)  # Decay rate to reach 0.05 in 8.333 hours
        return np.exp(-decay_rate * time)

    def growth_curve(time):
        # Growth rate to reach 1.0 in 8.33333 hours (no recording longer than 8.333 hours)
        growth_rate = 1.0 / (8.33333 * 3600)
        return growth_rate * time

    def write_circadian_features(subject_id, cosine_feature, decay_feature, growth_feature):
        cosine_path = Constants.ALTINI_FEATURE_FILE_PATH.joinpath(subject_id + '_cosine_feature.out')
        decay_path = Constants.ALTINI_FEATURE_FILE_PATH.joinpath(subject_id + '_decay_feature.out')
        growth_path = Constants.ALTINI_FEATURE_FILE_PATH.joinpath(subject_id + '_growth_feature.out')

        np.savetxt(cosine_path, cosine_feature, fmt='%f')
        np.savetxt(decay_path, decay_feature, fmt='%f')
        np.savetxt(growth_path, growth_feature, fmt='%f')

        return

    def build_from_accel(subject_id, valid_epochs):
        cropped_motion_path = MotionService.get_cropped_file_path(subject_id)
        motion_4d_array = pd.read_csv(str(cropped_motion_path), delimiter=' ').values

        FeatureBuilder.build_write_accel_features(motion_4d_array, valid_epochs, subject_id)

        FeatureBuilder.build_write_bcg_features(motion_4d_array, valid_epochs, subject_id)

        return

    def build_write_bcg_features(motion_4d_array, valid_epochs, subject_id):

        n_features=11
        features_all_epochs = np.empty((0, n_features))
        for epoch in valid_epochs:
            print(epoch.timestamp)
            # TODO: Timestamps will have to be more thought through
            start_time = epoch.timestamp - 135 #5min window
            end_time = epoch.timestamp + 30 + 135 #5min window

            mask = (motion_4d_array[:, 0] >= start_time) & (motion_4d_array[:, 0] < end_time)
            motion_snippet = motion_4d_array[mask]

            dif_array = np.diff(np.asarray(motion_snippet[:, 0]))
            freq_array = 1 / dif_array
            accel_fs = round(statistics.mode(freq_array) * 2) / 2

            # TODO: better error checking
            accel_snip = np.asarray(motion_snippet[:,1])
            mvmt_snip = Proell.detect_movements(accel_snip, accel_fs)
            accel_norm = (accel_snip - np.mean(accel_snip)) / np.std(accel_snip)
            coarse_peaks, bcg_enhanced, bcg_coarse, ijk = Proell.segmenter(start_time, accel_norm, accel_fs, show=False)
            j_peaks = [array[1] for array in ijk]  # NOTE: not currently used. using coarse peaks instead
            clean_j = Proell.reject_mvmt_peaks(coarse_peaks, mvmt_snip, mvmtDistance=10)

            bcg_hr_temp, z_checked_difs, z_checked_ts_seconds = Proell.heartrate_from_indices(clean_j, accel_fs)
            z_checked_difs = np.array(z_checked_difs)

            #TODO: more peak checking
            sd = np.diff(z_checked_difs) # successive differences
            rMSSD = np.sqrt(np.mean(np.square(sd)))/accel_fs*1e3 #scaling doesn't matter, we'll normalize later
            SDNN = np.std(z_checked_difs)/accel_fs*1e3
            pnn50 = FeatureBuilder.calc_pnn50(z_checked_difs, accel_fs)

            #interpolate difs to 100 hz
            new_timestamps = np.arange(z_checked_ts_seconds[0], z_checked_ts_seconds[-1], 0.01) # 100 Hz
            interpolated_z_checked_diffs = np.interp(new_timestamps, z_checked_ts_seconds, z_checked_difs)

            lf_power, hf_power, lf_peak_freq, hf_peak_freq, total_power = FeatureBuilder.welches_psd_features(interpolated_z_checked_diffs, 100)
            normalized_power = total_power #duplicating so I don't forget. It'll get normalized later

            #TODO: breathing rate
            breathing_rate = FeatureBuilder.get_breathing_rate(motion_snippet, accel_fs)

            features_one_epoch = np.array([bcg_hr_temp, rMSSD, SDNN, pnn50, lf_power, hf_power, lf_peak_freq,
                                            hf_peak_freq, total_power, normalized_power, breathing_rate])

            features_all_epochs = np.vstack((features_all_epochs, features_one_epoch))


        # Normalize the features
        normalized_total_features = np.zeros_like(features_all_epochs)

        for i in range(len(features_one_epoch)):
            feature_column = features_all_epochs[:, i]
            if i == 8:
                normalized_total_features[:, i] = feature_column  # No normalization for the 9th column
            else:
                normalized_column = FeatureBuilder.normalize_feature(feature_column)
                normalized_total_features[:, i] = normalized_column

        bcg_feature_path = Constants.ALTINI_FEATURE_FILE_PATH.joinpath(subject_id + '_bcg_features.out')
        np.savetxt(bcg_feature_path, normalized_total_features, fmt='%f')

        return
    def get_breathing_rate(motion_snippet, accel_fs):
        avg_filter_window_size = int(1.51515 * accel_fs)
        averaging_filter = np.ones(avg_filter_window_size) / avg_filter_window_size
        freq_range = (0.13, 0.66)

        timestamps = motion_snippet[:,0]
        accel_x = motion_snippet[:,1]
        accel_y = motion_snippet[:, 2]
        accel_z = motion_snippet[:, 3]

        chunk_size = 30

        resp_rates_aggregate = []

        start_time_range = int(timestamps[0])
        end_time_range = int(math.ceil(timestamps[-1]+1))

        for start_time in range(start_time_range, end_time_range - chunk_size + 1, chunk_size):
            end_time = start_time + chunk_size
            mask = (timestamps >= start_time) & (timestamps < end_time)
            motion_chunk = motion_snippet[mask]

            accel_x_snip = motion_chunk[:, 1]
            accel_y_snip = motion_chunk[:, 2]
            accel_z_snip = motion_chunk[:, 3]

            motion_x = Proell.detect_movements(accel_x_snip, accel_fs)
            motion_y = Proell.detect_movements(accel_y_snip, accel_fs)
            motion_z = Proell.detect_movements(accel_z_snip, accel_fs)

            if sum(motion_x) == 0:
                accel_x_normal = FeatureBuilder.normalize_array(accel_x_snip)
                resp_wave_x = convolve(accel_x_normal, averaging_filter, mode='same')
                resp_rate_x = FeatureBuilder.return_resp_rate(resp_wave_x, freq_range, accel_fs)
                if resp_rate_x >= 8.1:
                    resp_rate_x = resp_rate_x
                else:
                    resp_rate_x = np.nan
            else:
                resp_rate_x = np.nan

            if sum(motion_y) == 0:
                accel_y_normal = FeatureBuilder.normalize_array(accel_y_snip)
                resp_wave_y = convolve(accel_y_normal, averaging_filter, mode='same')
                resp_rate_y = FeatureBuilder.return_resp_rate(resp_wave_y, freq_range, accel_fs)
                if resp_rate_y >= 8.1:
                    resp_rate_y = resp_rate_y
                else:
                    resp_rate_y = np.nan
            else:
                resp_rate_y = np.nan

            if sum(motion_z) == 0:
                accel_z_normal = FeatureBuilder.normalize_array(accel_z_snip)
                resp_wave_z = convolve(accel_z_normal, averaging_filter, mode='same')
                resp_rate_z = FeatureBuilder.return_resp_rate(resp_wave_z, freq_range, accel_fs)
                if resp_rate_z >= 8.1:
                    resp_rate_z = resp_rate_z
                else:
                    resp_rate_z = np.nan
            else:
                resp_rate_z = np.nan

            resp_rate_aggregate = FeatureBuilder.calculate_final_resp(resp_rate_x, resp_rate_y, resp_rate_z)
            resp_rates_aggregate.append(resp_rate_aggregate)

        # Calculate overall 5 min array mean (or nan)
        mean_without_nan = np.nanmean(resp_rates_aggregate)
        if np.isnan(mean_without_nan): #if no valid HRs, just set it to 16 (avg btwn 12-20)
            resp_rate_final = 16.0
        else:
            resp_rate_final = mean_without_nan

        return resp_rate_final


    def normalize_array(window):  # Normalize each component independently with z-scores
        window_mean = np.mean(window)
        window_std = np.std(window)
        window_normalized = (window - window_mean) / window_std

        return window_normalized

    def return_resp_rate(array, freq_range, fs):
        # Compute FFT for each component
        freq_samples = np.fft.fftfreq(len(array), 1 / fs)
        mask = (freq_samples >= freq_range[0]) & (freq_samples <= freq_range[1])

        fft_result = fft(array)
        fft_result = fft_result[mask]

        # Identify the frequency with the highest amplitude response and calculate the respiratory rate
        max_freq = np.argmax(np.abs(fft_result))

        resp_rate_x = freq_samples[mask][max_freq] * 60

        return resp_rate_x

    def calculate_final_resp(resp_rate_x, resp_rate_y, resp_rate_z):
        if np.isnan(resp_rate_x) and np.isnan(resp_rate_y) and np.isnan(resp_rate_z):
            final_resp = np.nan
        elif np.isnan(resp_rate_x) and np.isnan(resp_rate_y):
            final_resp = resp_rate_z
        elif np.isnan(resp_rate_x) and np.isnan(resp_rate_z):
            final_resp = resp_rate_y
        elif np.isnan(resp_rate_y) and np.isnan(resp_rate_z):
            final_resp = resp_rate_x
        elif np.isnan(resp_rate_x):
            final_resp = 0.5 * (resp_rate_y + resp_rate_z)
        elif np.isnan(resp_rate_y):
            final_resp = 0.5 * (resp_rate_x + resp_rate_z)
        elif np.isnan(resp_rate_z):
            final_resp = 0.5 * (resp_rate_x + resp_rate_y)
        else:
            dif_xy = np.abs(resp_rate_x - resp_rate_y)
            dif_xz = np.abs(resp_rate_x - resp_rate_z)
            dif_yz = np.abs(resp_rate_y - resp_rate_z)

            min_diff = min(dif_xy, dif_xz, dif_yz)

            if min_diff == dif_xy:
                final_resp = 0.5 * (resp_rate_x + resp_rate_y)
            elif min_diff == dif_xz:
                final_resp = 0.5 * (resp_rate_x + resp_rate_z)
            elif min_diff == dif_yz:
                final_resp = 0.5 * (resp_rate_y + resp_rate_z)

        return final_resp


    def welches_psd_features(signal, fs):
        # Calculate the power spectral density using Welch's method
        # f, Pxx = welch(signal, fs=fs, nperseg=64)  # You may adjust nperseg as needed
        signal = signal - np.mean(signal)
        min_frequency = 0.01
        nperseg = int((2 / min_frequency) * fs)
        nfft = int(nperseg * 2)

        f, Pxx = welch(
            signal,
            fs=fs,
            scaling="density",
            detrend=False,
            nfft=nfft,
            average="mean",
            nperseg=nperseg,
            window="hann",
        )

        # Find the indices corresponding to the LF frequency range (0.04-0.15 Hz)
        lf_indices = np.where((f >= 0.04) & (f <= 0.15))

        # Find the indices corresponding to the HF frequency range (0.15-0.4 Hz)
        hf_indices = np.where((f >= 0.15) & (f <= 0.4))

        # Integrate the power within the LF and HF frequency ranges
        lf_power = np.trapz(Pxx[lf_indices], f[lf_indices])
        hf_power = np.trapz(Pxx[hf_indices], f[hf_indices])

        # Find the LF peak frequency
        lf_peak_freq_indices, _ = find_peaks(Pxx[lf_indices], height=0)
        if len(lf_peak_freq_indices) > 0:
            lf_peak_freq = f[lf_indices][lf_peak_freq_indices[0]]
        else:
            lf_peak_freq = 0

        # Find the HR peak frequency
        hf_peak_freq_indices, _ = find_peaks(Pxx[hf_indices], height=0)
        if len(hf_peak_freq_indices) > 0:
            hf_peak_freq = f[hf_indices][hf_peak_freq_indices[0]]
        else:
            hf_peak_freq = 0

        total_power = np.trapz(Pxx, f)

        return lf_power, hf_power, lf_peak_freq, hf_peak_freq, total_power



    def calc_pnn50(nn_intervals, accel_fs):
        nn_intervals_ms = nn_intervals / accel_fs * 1000
        sd_ms = abs(np.diff(nn_intervals_ms))

        nn50_count = np.sum(sd_ms > 50)

        # Step 3: Calculate pNN50 as a percentage
        pnn50 = (nn50_count / len(nn_intervals)) * 100
        return pnn50

    def build_write_accel_features(motion_4d_array, valid_epochs, subject_id):
        window_length = 30

        features_all_epochs = []

        for epoch in valid_epochs:
            start_time = epoch.timestamp
            end_time = epoch.timestamp + window_length

            # create mask based on timestamps (is this quicker than pandas dataframe? who's to say)
            mask = (motion_4d_array[:, 0] >= start_time) & (motion_4d_array[:, 0] < end_time)
            # Use the mask to extract the desired rows
            motion_snippet = motion_4d_array[mask]

            dif_array = np.diff(np.asarray(motion_snippet[:, 0]))
            freq_array = 1 / dif_array
            accel_fs = round(statistics.mode(freq_array) * 2) / 2

            # First: filtered snippet features
            filtered_snippet = np.column_stack([FeatureBuilder.absolute_butter(motion_snippet[:, i], accel_fs) for i in range(1,4)])
            filtered_trimmed_means = [FeatureBuilder.trim_and_mean(filtered_snippet[:, i], sixMode=False) for i in range(filtered_snippet.shape[1])]
            filtered_maxes = [max(filtered_snippet[:, i]) for i in range(filtered_snippet.shape[1])]
            filtered_iqr = [FeatureBuilder.get_iqr(filtered_snippet[:, i]) for i in range(filtered_snippet.shape[1])]

            # 2/3: calculate MAD & Arm Angle @ 5s Epochs
            mad_all = []
            angle_all = []
            for index_5s in range(6):
                mask_5s = (motion_4d_array[:, 0] >= start_time + 5*index_5s) & (motion_4d_array[:, 0] < start_time + 5*(1+index_5s))
                motion_snippet_5s = motion_4d_array[mask_5s]

                mad_5s = [FeatureBuilder.calc_MAD_5s(motion_snippet_5s[:,i]) for i in range(1,4)]
                mad_all.append(mad_5s)

                angle_5s = FeatureBuilder.calculate_arm_angle(motion_snippet_5s[:,[1,2,3]])
                angle_all.append(angle_5s)

            mad_all = np.array(mad_all)

            # SECOND: MAD Features
            mad_trimmed_means = [FeatureBuilder.trim_and_mean(mad_all[:, i], sixMode=True) for i in range(mad_all.shape[1])]
            mad_max_fury_road = [max(mad_all[:, i]) for i in range(mad_all.shape[1])]
            mad_iqr = [FeatureBuilder.get_iqr(mad_all[:, i]) for i in range(mad_all.shape[1])]

            # Third: Arm Angle Features
            angle_trimmed_means = FeatureBuilder.trim_and_mean(angle_all, sixMode=True)
            angle_max = max(angle_all)
            angle_iqr = FeatureBuilder.get_iqr(angle_all)

            features_one_epoch = np.concatenate((filtered_trimmed_means, filtered_maxes, filtered_iqr, mad_trimmed_means, mad_max_fury_road,
                                           mad_iqr, [angle_trimmed_means, angle_max, angle_iqr]))

            features_all_epochs.append(features_one_epoch)
        features_all_epochs = np.vstack(features_all_epochs)

        normalized_total_features = np.zeros_like(features_all_epochs)

        for i in range(21):
            feature_column = features_all_epochs[:, i]
            normalized_column = FeatureBuilder.normalize_feature(feature_column)
            normalized_total_features[:, i] = normalized_column

        accel_feature_path = Constants.ALTINI_FEATURE_FILE_PATH.joinpath(subject_id + '_accel_features.out')
        np.savetxt(accel_feature_path, normalized_total_features, fmt='%f')

        return

    def normalize_feature(feature_column):
        # Calculate the 5th and 95th percentiles
        percentile_5 = np.percentile(feature_column, 5)
        percentile_95 = np.percentile(feature_column, 95)

        # Calculate the range for normalization
        range_ = percentile_95 - percentile_5

        # Normalize the feature column to the range [0, 1]
        normalized_column = (feature_column - percentile_5) / range_

        return normalized_column
    def calculate_arm_angle(data):
        # Calculate the medians of x, y, and z columns
        x_median = np.median(data[:, 0])
        y_median = np.median(data[:, 1])
        z_median = np.median(data[:, 2])

        # Calculate the angle in radians
        angle_rad = np.arctan2(z_median, np.sqrt(x_median ** 2 + y_median ** 2))

        # Convert the angle to degrees
        angle_deg = angle_rad * 180 / np.pi

        return angle_deg

    def calc_MAD_5s(array):
        mad = np.mean(np.abs(array - np.mean(array)))
        return mad

    def absolute_butter(data, fs, lowcut=3.0, highcut=11.0,  order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        filtered_data = scipy.signal.lfilter(b, a, data)
        abs_butter = np.abs(filtered_data)
        return abs_butter

    def trim_and_mean(array, sixMode):

        # Sort the array in ascending order
        sorted_array = np.sort(array)

        if sixMode:
            trimmed_array = sorted_array[1:-1]
        else:
            # Calculate the number of elements to trim from the top and bottom
            trim_size = int(len(array) * 0.1)

            # Trim the top and bottom 10% of elements
            trimmed_array = sorted_array[trim_size:-trim_size]

        # Calculate the mean of the remaining elements
        mean_value = np.mean(trimmed_array)

        return mean_value

    def get_iqr(array):
        # Calculate the 25th and 75th percentiles
        q1 = np.percentile(array, 25)
        q3 = np.percentile(array, 75)

        # Calculate the IQR (Interquartile Range)
        iqr = q3 - q1

        return iqr





