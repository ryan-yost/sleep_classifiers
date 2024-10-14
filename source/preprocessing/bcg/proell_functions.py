import numpy as np
import pandas as pd
import scipy.signal as sgnl


class Proell:
    def reject_mvmt_peaks(peaks, mvmt, mvmtDistance=10):
        clean_peaks = []

        for peak in peaks:
            start_index = max(0, peak - mvmtDistance)
            end_index = min(len(mvmt), peak + mvmtDistance + 1)

            if not any(mvmt[start_index:end_index]):
                clean_peaks.append(peak)

        return clean_peaks


    def heartrate_from_indices(indices, f, max_std_seconds=float("inf"),
                               min_num_peaks=2, use_median=False, minHR=30, maxHR=200):
        """Calculate heart rate from given peak indices

        Args:
            indices (`np.ndarray`): indices of detected peaks
            f (float): in Hz; sampling rate of BCG signal
            min_num_peaks (int): minimum number of peaks to consider valid
            max_std_seconds (float): in seconds; maximum standard deviation
                of peak distances
            use_median (bool): calculate heart rate with median instead of
                mean

        Returns:
            float: mean heartrate estimation in beat/min
        """

        if len(indices) < min_num_peaks:
            hr = -1

        diffs = np.diff(indices)
        timestamps = indices[1:]
        nn_std = np.std(diffs)

        maxDistance = 60*f/minHR
        minDistance = 60*f/maxHR

        hr_checked_diffs = [] #within 30-200 bpm
        hr_checked_ts = []
        for i, diff in enumerate(diffs):
            if minDistance <= diff <= maxDistance:
                hr_checked_diffs.append(diff)
                hr_checked_ts.append(timestamps[i])


        # check z-score for each peak to remove outliers
        mean_diffs = np.mean(hr_checked_diffs)
        std_deviation_diffs = np.std(hr_checked_diffs)
        minZ, maxZ = -1.3, 1.3

        z_checked_diffs = []
        z_checked_ts = []

        for i, diff in enumerate(hr_checked_diffs):
            if minZ <= (diff - mean_diffs) / std_deviation_diffs <= maxZ:
                z_checked_diffs.append(diff)
                z_checked_ts.append(hr_checked_ts[i])

        if nn_std > max_std_seconds:
            hr = -1

        if use_median:
            hr = 60. * f / np.median(z_checked_diffs)
        else:
            hr = 60. * f / np.mean(z_checked_diffs)

        #print(z_checked_diffs)
        z_checked_ts_seconds = np.array(z_checked_ts) / f


        return hr, z_checked_diffs, z_checked_ts_seconds


    def get_heartrate_pipe(segmenter, max_std_seconds=float("inf"), min_num_peaks=2,
                           use_median=False, index=None):
        """build function that estimates heart rate from detected peaks in
        input signal

        If stddev of peak distances exceeds `max_std_seconds` or less than
        `min_.num_peaks` peaks are found, input signal is marked as invalid
        by returning -1.
        If the `segmenter` returns tuples of wave indices (e.g. IJK instead
        of just J) the wave used for calculations has to be specified with
        `index`.

        Args:
            segmenter (function): BCG segmentation algorithm
            max_std_seconds (float): maximum stddev of peak distances
            min_num_peaks (int): minimum number of detected peaks
            use_median (bool): calculate heart rate from median of peak
                distances instead of mean
            index (int): index of wave used for calculations

        Returns:
            `function`: full heart rate estimation algorithm
        """

        def pipe(x, f, **args):
            indices = segmenter(x, f, **args)
            if index is not None:
                indices = indices[:, index]
            hr = Proell.heartrate_from_indices(indices, f,
                                        max_std_seconds=max_std_seconds,
                                        min_num_peaks=min_num_peaks,
                                        use_median=use_median)
            return hr

        return pipe


    def get_heartrate_score_pipe(segmenter, use_median=False, index=None):
        """build function that estimates heart rate from detected peaks in
        input signal and return both heart rate and stddev of peak distances

        If the `segmenter` returns tuples of wave indices (e.g. IJK instead
        of just J) the wave used for calculations has to be specified with
        `index`.

        Args:
            segmenter (function): BCG segmentation algorithm
            use_median (bool): calculate heart rate from median of peak
                distances instead of mean
            index (int): index of wave used for calculations

        Returns:
            `function`: full heart rate estimation algorithm that returns
            both estimated heart rate and stddev of peak distances for given
            signal
        """

        def pipe(x, f, **args):
            indices = segmenter(x, f, **args)
            if index is not None:
                indices = indices[:, index]
            n = len(indices)
            if n < 2:
                return -1, -1
            diffs_std = np.std(np.diff(indices) / f)
            hr = Proell.heartrate_from_indices(indices, f, max_std_seconds=float("inf"),
                                        min_num_peaks=2, use_median=use_median)

            return hr, diffs_std

        return pipe

    """BCG segmetation algorithm
    """

    import numpy as np
    import pandas as pd
    import scipy.signal as sgnl


    def enhance_signal(x, f, f1=2., f2=10., f3=5., order=2):
        """Preprocess BCG and enhance ejection wave amplitudes by cubing the
        signal and computing filtered second derivative

        Args:
            x (`1d array-like`): raw BCG signal
            f (float): in Hz; sampling rate of input signal
            f1 (float): in Hz; lower cutoff frequency of bandpass filter
            f2 (float): in Hz; higher cutoff frequency of bandpass filter
            f3 (float): in Hz; cutoff frequency of lowpass filter for cubed
                signal
            order (int): order of Butterworth filters

        Returns:
            `(1d array, 1d array)`: bandpass-filtered and enhanced signals
        """

        coeffs1 = sgnl.butter(N=order, Wn=np.divide([f1, f2], f / 2.),
                              btype="bandpass")
        coeffs2 = sgnl.butter(N=order, Wn=np.divide(f3, f / 2.), btype="lowpass")

        # basic preprocessing
        x_filt = sgnl.filtfilt(coeffs1[0], coeffs1[1], x)

        # IJK enhancement
        x_enhanced = sgnl.filtfilt(coeffs2[0], coeffs2[1], x_filt ** 3)
        x_enhanced = -np.gradient(np.gradient(x_enhanced))

        return x_filt, x_enhanced


    def renormalize_signal(x, f, window_length=1.):
        """Re-normalize signal by division with a moving stddev signal

        Args:
            x (`1d array-like`): input signal
            f (float): in Hz; sampling rate of input signal
            window_length (float): in seconds; window length for moving
                stddev calculation

        Returns:
            `1d-array`: re-normalized signal
        """

        rolling_std = pd.Series(x).rolling(int(f * window_length),
                                           center=True,
                                           min_periods=1).std()

        rolling_std[rolling_std == 0] = 1.  # avoid divide by zero
        x = np.divide(x, rolling_std)

        return x


    def get_coarse_signal(x, f, cutoff=1.5, order=4):
        """Calculate coarse BCG signal from enhanced BCG

        Args:
            x (`1d array-like`): enhanced BCG signal
            f (float): in Hz; sampling rate of input signal
            cutoff (float): in Hz; cutoff frequency of lowpass filter
            order (int): order of Butterworth filter

        Returns:
            `1d array`: coarse BCG signal
        """

        coeffs = sgnl.butter(N=order, Wn=np.divide(cutoff, f / 2.), btype="lowpass")

        return sgnl.filtfilt(coeffs[0], coeffs[1], np.abs(x))


    def find_ijk(x, f, ws, wave_dist=0.045):
        """Find IJK indices in small window around peaks in coarse signal

        Args:
            x (`1d array-like`): BCG signal
            f (float): in Hz; sampling rate of BCG signal
            ws (`tuple(float)`): weights for waves
            wave_dist (float): in seconds; minimum distance between peaks

        Returns:
            `1d array`: locations of one peak tuple
        """

        n = len(ws)
        waves = sgnl.find_peaks(np.abs(x), distance=int(wave_dist*f))[0]
        if len(waves) < n:
            return None
        maxi = np.argmax(np.correlate(x[waves], ws, mode="valid"))
        return np.asarray(waves)[np.arange(n) + maxi]


    def refine_ijk(x, f, ijk, ws, window_length=0.045):
        """Refine IJK locations detected in enhanced signal

        Args:
            x (`1d array-like`): preprocessed BCG signal
            f (float): in Hz; sampling rate of input signal
            ijk (`tuple(int)`): I, J and K indices
            ws (`tuple(float)`): weights for each wave
            window_length (float): in seconds; small window length for peak
                refinement

        Returns:
            `tuple(int)`: refined IJK peaks
        """

        winsize = int(window_length * f)
        for i, p in enumerate(ijk):
            win = np.sign(ws[i]) * Proell.get_padded_window(x, p, winsize,
                                                     padding_value=np.nan)
            maxi = np.nanargmax(win)
            ijk[i] = p - np.ceil(winsize / 2).astype(int) + maxi

        return ijk


    def segmenter(time, x, f, f1=2., f2=10., f3=5., f4=1.35, order=2, renorm_window=1.,
                  coarse_dist=0.3, coarse_window=0.5, ws=(-1, 1, -1),
                  refine_window=0.1, coarse_order=3, renorm=True, refine=True, show=True):
        """BCG segmentation algorithm

        With (-1, 1, -1) the algorithm searches for a valley-peak-valley
        sequence in the BCG signal around coarse BCG locations.
        The valley-peak-valley sequence likely corresponds to I, J and K
        waves.  By using different weights, one could search for more or
        fewer waves (coarse window should be adjusted accordingly).

        Args:
            x (`1d array-like`): raw BCG signal
            f (float): in Hz; sampling rate of input signal
            f1 (float): in Hz; lower cutoff frequency of bandpass filter
            f2 (float): in Hz; higher cutoff frequency of bandpass filter
            f3 (float): in Hz; cutoff frequency of lowpass filter (enhanced
                BCG calculation)
            f4 (float):in Hz; cutoff frequency for lowpass filter (coarse
                BCG calculation)
            order (int): order of Butterworth filters in BCG enhancement
            renorm_window (float): in seconds; window size for signal
                re-normalization
            coarse_dist (float): in seconds; minimum distance of coarse
                peaks
            coarse_window (float): in seconds; window length for detection
                of I, J and K peaks around coarse locations
            ws (`tuple(float)`): weights for weighted sum calculation
            refine_window (float): in seconds; small window length for peak
                refinement
            coarse_order (int): order of Butterworth lowpass for coarse BCG
                calculation
            renorm (bool): apply re-normalization of enhanced signal
            refine (bool): apply peak refinement

        Returns:
            `array`: n x m array, with n being the number of detected
            peak complexes and m being the number of waves per complex
            (length of `ws`)
        """

        x_filt, x_enhanced = Proell.enhance_signal(x, f, f1=f1, f2=f2, f3=f3, order=order)
        if renorm:
            x_enhanced = Proell.renormalize_signal(x_enhanced, f,
                                            window_length=renorm_window)
        x_coarse = Proell.get_coarse_signal(x_enhanced, f, cutoff=f4, order=coarse_order)

        coarse_indices = sgnl.find_peaks(x_coarse, distance=int(coarse_dist*f))[0]

        window_size = int(coarse_window * f)
        ijk_indices = []
        for ci in coarse_indices:
            win = Proell.get_padded_window(x_enhanced, ci, window_size)
            ijk = Proell.find_ijk(win, f, ws=ws, wave_dist=1./f)
            if ijk is None:
                win = Proell.get_padded_window(x_filt, ci, window_size)
                ijk = Proell.find_ijk(win, f, ws=ws, wave_dist=1./f)
            if ijk is None:
                continue

            ijk_indices.append(np.asarray(ijk) + max(0, ci-np.ceil(window_size/2.)))


        ijk_refined = []
        if refine:
            for i in range(len(ijk_indices)):
                ijk_refined_temp = Proell.refine_ijk(x_filt, f, ijk_indices[i].astype(int),
                                            ws=ws, window_length=refine_window)
                ijk_refined.append(ijk_refined_temp)

        ijk_final = np.array(ijk_indices).astype(int)

        return coarse_indices, x_enhanced, x_coarse, ijk_refined

    def detect_movements(x, f, f1=2., f2=10., th0=125., thf1=2.5, order=2,
                         percentile=90, stdwin=2., th2=3., margin=1.):
        """Detect movements in raw BCG signal by simple applying simple
        thresholds

        Every data point in the input signal is checked against different
        thresholds and bits corresponding to different thresholds are set if
        the values exceed ceratin thresholds.
        If the value of the movement signal is greater 0, movements were
        detected at that point.

        Applied thresholds are:

           - raw amplitude: signal should not exceed `th0`
           - deviation: bandpass-filtered signal should not exceed `thf1`
             times the `percentile`th percentile of the bandpass-filtered
             signal
           - moving deviation: moving stddev with window size `stdwin` of
             bandpass-filtered signal should not exceed `th2`

        After thresholding, a moving maximum calculation is applied to
        provide a 'safety margin' for detected movements.

        Args:
            x (`1d array-like`): raw BCG signal
            f (float): in Hz; sampling rate of input signal
            f1 (float): in Hz; lower cutoff frequency of bandpass filter
            f2 (float): in Hz; higher cutoff frequency of bandpass filter
            th0 (float): maximum raw amplitude value
            thf1 (float): thresholding factor for deviation threshold
            order (int): order of Butterworth bandpass filter
            percentile (float): percentile used for deviation thresholding
            stdwin (float): in seconds; window length for moving stddev
                calculation
            th2 (float): moving deviation threshold
            margin (float): in seconds; window size for moving maximum
                calculation of safety margin.  `margin` is applied to both
                sides of detected movements.

        Returns:
            `1d array`: movement signal containing values greater 0 where
            movements were detected
        """

        xfilt = Proell.filter_bandpass(x, f, cutoff1=f1, cutoff2=f2, order=order)
        xstd = pd.Series(xfilt).rolling(int(f*stdwin), center=True, min_periods=1
                                        ).std()

        # check all four thresholds
        raw_amplitude_bit = np.greater(np.abs(x), th0) * 1
        deviation_bit = np.greater(np.abs(xfilt),
                                   thf1 * np.percentile(np.abs(xfilt), percentile)
                                   ) * 2
        moving_deviation_bit = np.greater(xstd.values, th2) * 4

        # combine thresholds to single movement signal
        movement_signal = np.zeros_like(x, dtype=int)
        for bits in [raw_amplitude_bit, deviation_bit, moving_deviation_bit]:
            movement_signal = np.bitwise_or(movement_signal, bits.astype(int))

        # apply 'safety' margin
        return pd.Series(movement_signal).rolling(int(2*f*margin), center=True,
                                                  min_periods=1
                                                  ).max().values


    def filter_bandpass(x, f, cutoff1, cutoff2, order=2):
        """Filter signal forwards and backwards with Butterworth bandpass
        """
        coeffs = sgnl.butter(N=order, Wn=np.divide([cutoff1, cutoff2],
                                                   f/2.), btype="bandpass")

        return sgnl.filtfilt(coeffs[0], coeffs[1], x)


    def filter_lowpass(x, f, cutoff, order=2):
        """Filter signal forwards and backwards with Butterworth lowpass
        """

        coeffs = sgnl.butter(N=order, Wn=np.divide(cutoff, f/2.),
                             btype="lowpass")

        return sgnl.filtfilt(coeffs[0], coeffs[1], x)


    def get_padded_window(x, i, n, nafter=None, padding_value=0.):
        """Get padded window from signal at specified location

        Args:
            x (`np.ndarray`): signal from which to extract window
            i (int): index location of window
            n (int): window size in samples (if asymmetric: number of
                samples before specified index)
            nafter (int): number of samples after specified index if
                asymmetric window
            padding_value (float or list): value used for padding at start
                and end of signal.  If None, nearest neighbor is used.

        Returns:
            `np.ndarray`: padded window within signal
        """

        x = np.asarray(x)

        # check inputs
        if i < 0 or i >= len(x):
            raise IndexError("Index %d out of range for array with length %d" %
                             (i, len(x)))
        if len(x.shape) not in {1, 2}:
            raise ValueError("x has to be 1- or 2-dimensional")
        if (len(x.shape) == 2 and hasattr(padding_value, "__len__") and
                len(padding_value) != x.shape[1]):
            raise ValueError("padding_value must be a single float or of same "
                             "length as x")

        # calculate left and right margins
        if not nafter:
            nbefore = int(np.ceil(n/2.))
            nafter = int(n // 2)
        else:
            nbefore = n

        # determine left and right padding values
        if padding_value is None:
            left_pad = x[max(0, i-nbefore)]
            right_pad = x[min(i+nafter, len(x))-1]
        else:
            left_pad = right_pad = padding_value
        left_pad = [left_pad]*max(0, nbefore - i)
        right_pad = [right_pad]*max(0, i+nafter-len(x))
        if len(x.shape) == 2:
            left_pad = np.reshape(left_pad, (-1, x.shape[1]))
            right_pad = np.reshape(right_pad, (-1, x.shape[1]))

        return np.concatenate([left_pad,
                               x[max(0, i-nbefore):min(i+nafter, len(x))],
                               right_pad])


    def data_from_uint16(data, factor=240.,
                         offset=2**15/240.):
        """Convert input data from uint16 format to Pascal

        Args:
            data (`1d array-like`): input data in uint16
            factor (float): data is divided by `factor` to convert raw
                values to Pascal
            offset (float): offset correcting unsigned values to signed Pa

        Returns:
            `1d array`: converted signal
        """

        return np.subtract(np.divide(data, factor), offset)