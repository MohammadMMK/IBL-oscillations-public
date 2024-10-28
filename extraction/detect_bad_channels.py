
# main author: Romain Ligneul 
# co-authors: Mohammad Keshtkar

import numpy as np
import scipy.signal
import scipy.fft
import ibldsp.utils as utils
import scipy

def detect_bad_channels(raw, fs, similarity_threshold=(-0.5, 1), psd_hf_threshold=None):
    """
    Bad channels detection for Neuropixel probes
    Labels channels
    0: all clear
    1: dead low coherence / amplitude
    2: noisy
    3: outside of the brain
    :param raw: [nc, ns]
    :param fs: sampling frequency
    :param similarity_threshold:
    :param psd_hf_threshold:
    :return: labels (numpy vector [nc]), xfeats: dictionary of features [nc]
    """

    def rneighbours(raw, n=1):  # noqa
        """
        Computes Pearson correlation with the sum of neighbouring traces
        :param raw: nc, ns
        :param n:
        :return:
        """
        nc = raw.shape[0]
        mixer = np.triu(np.ones((nc, nc)), 1) - np.triu(np.ones((nc, nc)), 1 + n)
        mixer += np.tril(np.ones((nc, nc)), -1) - np.tril(np.ones((nc, nc)), -n - 1)
        r = rcoeff(raw, np.matmul(raw.T, mixer).T)
        r[np.isnan(r)] = 0
        return r

    def detrend(x, nmed):
        """
        Subtract the trend from a vector
        The trend is a median filtered version of the said vector with tapering
        :param x: input vector
        :param nmed: number of points of the median filter
        :return: np.array
        """
        ntap = int(np.ceil(nmed / 2))
        xf = np.r_[np.zeros(ntap) + x[0], x, np.zeros(ntap) + x[-1]]
        # assert np.all(xcorf[ntap:-ntap] == xcor)
        xf = scipy.signal.medfilt(xf, nmed)[ntap:-ntap]
        return x - xf

    def channels_similarity(raw, nmed=0):
        """
        Computes the similarity based on zero-lag crosscorrelation of each channel with the median
        trace referencing
        :param raw: [nc, ns]
        :param nmed:
        :return:
        """

        def fxcor(x, y):
            return scipy.fft.irfft(
                scipy.fft.rfft(x) * np.conj(scipy.fft.rfft(y)), n=raw.shape[-1]
            )

        def nxcor(x, ref):
            ref = ref - np.mean(ref)
            apeak = fxcor(ref, ref)[0]
            x = x - np.mean(x, axis=-1)[:, np.newaxis]  # remove DC component
            return fxcor(x, ref)[:, 0] / apeak

        ref = np.median(raw, axis=0)
        xcor = nxcor(raw, ref)

        if nmed > 0:
            xcor = detrend(xcor, nmed) + 1
        return xcor

    nc, _ = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    xcor = channels_similarity(raw)
    fscale, psd = scipy.signal.welch(raw * 1e6, fs=fs)  # units; uV ** 2 / Hz
    if psd_hf_threshold is None:
        # the LFP band data is obviously much stronger so auto-adjust the default threshold
        psd_hf_threshold = 1.4 if fs < 5000 else 0.02
        sos_hp = scipy.signal.butter(
            **{"N": 3, "Wn": 20, "btype": "highpass", "fs": 500}, output="sos"
        )
    else:
        sos_hp = scipy.signal.butter(*0
            **{"N": 3, "Wn": 300 / fs * 2, "btype": "highpass"}, output="sos"
        )
    hf = scipy.signal.sosfiltfilt(sos_hp, raw)
    xcorf = channels_similarity(hf)

    xfeats = {
        "ind": np.arange(nc),
        "rms_raw": utils.rms(raw),  # very similar to the rms avfter butterworth filter
        "xcor_hf": detrend(xcor, 11),
        "xcor_lf": xcorf - detrend(xcorf, 11) - 1,
        "psd_hf": np.mean(psd[:, fscale > (fs / 2 * 0.8)], axis=-1),  # 80% nyquists
    }

    # make recommendation
    ichannels = np.zeros(nc)
    idead = np.where(similarity_threshold[0] > xfeats["xcor_hf"])[0]
    inoisy = np.where(
        np.logical_or(
            xfeats["psd_hf"] > psd_hf_threshold,
            xfeats["xcor_hf"] > similarity_threshold[1],
        )
    )[0]
    # the channels outside of the brains are the contiguous channels below the threshold on the trend coherency
    ioutside = np.where(xfeats["xcor_lf"] < -0.75)[0]
    if ioutside.size > 0 and ioutside[-1] == (nc - 1):
        a = np.cumsum(np.r_[0, np.diff(ioutside) - 1])
        ioutside = ioutside[a == np.max(a)]
        ichannels[ioutside] = 3

    # indices
    ichannels[idead] = 1
    ichannels[inoisy] = 2
    # from ibllib.plots.figures import ephys_bad_channels
    # ephys_bad_channels(x, 30000, ichannels, xfeats)
    return ichannels, xfeats