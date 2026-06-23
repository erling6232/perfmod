import numpy as np
import scipy.ndimage
import scipy.stats
import scipy.interpolate
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from imagedata import Series

from perfmod.fit_curves import show
import matplotlib.pyplot as plt


def logistic(t, A, K, B, M):
    return A + (K - A) / (1 + np.exp(-B * (t - M)))


def first_major_peak_ttp(t, E, post_idx_start=0, min_prominence=5.0, min_distance_seconds=None):
    """
    t: 1D array of acquisition times (same units as desired TTP)
    E: 1D array of percent-enhancement values (same length as t)
    post_idx_start: index to start searching (first post-contrast frame index)
    min_prominence: minimum peak prominence (same units as E, e.g., percent)
    min_distance_seconds: minimum separation between peaks (None to use 1 frame)
    Returns: dict with keys: peak_index, peak_time, peak_value, peaks_indices, properties
    """
    t = np.asarray(t)
    E = np.asarray(E)
    assert t.shape == E.shape

    # Restrict to post-contrast frames
    t_post = t[post_idx_start:]
    E_post = E[post_idx_start:]

    # Convert min_distance_seconds to samples if provided
    if min_distance_seconds is None:
        distance = 1
    else:
        # compute sample spacing (may be nonuniform -> use median)
        dt = np.median(np.diff(t_post))
        distance = max(1, int(np.round(min_distance_seconds / dt)))

    peaks, props = find_peaks(E_post, prominence=min_prominence, distance=distance)

    tries = 0
    while peaks.size == 0:
        tries += 1
        if tries > 100:
            return {"peak_index": None, "peak_time": None, "peak_value": None, "peaks_indices": peaks + post_idx_start, "properties": props}

        min_prominence *= 0.9
        peaks, props = find_peaks(E_post, prominence=min_prominence, distance=distance)

    # select first major peak (lowest index in time among detected peaks)
    first_peak_local = peaks[0]
    first_peak_global = first_peak_local + post_idx_start

    return {
        "peak_index": int(first_peak_global),
        "peak_time": float(t[first_peak_global]),
        "peak_value": float(E[first_peak_global]),
        "peaks_indices": (peaks + post_idx_start).tolist(),
        "properties": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in props.items()}
    }

# Example usage:
# t = np.array([0,10,20,30,40,50,60])            # seconds
# E = np.array([0,0,5,30,80,60,20])             # percent enhancement
# result = first_major_peak_ttp(t, E, post_idx_start=2, min_prominence=10, min_distance_seconds=15)
# print(result)



def compute_washin(
        img4d: Series,
        times: np.ndarray = None,
        metric: str | np.ndarray = 'percent',
        baseline_frames=5,
        smooth_window=3,
        arrival_thresh_pct=5.0,
        early_window_s=10.0,
        method='early',
        t_injection=None,
        r1=None,
        T10=None,
        TR=None, TE=None, FA_deg=None,
        vis=False,
        savefig=None
) -> tuple[Series, Series, Series, Series]:
    """Compute washin_map, peak_amp and ttp_map.

    Arguments:
        img4d: Series array, first axis must be time
        times: 1D array length nt, acquisition time (s) of each frame
        metric: precomputed metric array (img4d percent or concentration) OR string 'percent' to compute from raw img4d.
        If metric is a string, function expects img4d raw and will compute percent-change baseline like earlier.
        baseline_frames: number of initial frames to compute baseline (default: 5)
        smooth_window: temporal smoothing windows (frames), set 0 to disable (default: 3)
        arrival_thresh_pct: % signal change threshold to detect bolus arrival (default: 5.0)
        early_window_s: (default: 10.0)
        method: 'early' uses linear fit over early_window_s after bolus arrival.
                'arrival-to-peak' computes slope = (peak - arrival_value) / (t_peak - t_arrival).
        t_injection: optional bolus injection time (default: None)
        r1=None,  # relaxivity (s^-1 mM^-1) if using concentration conversion
        T10=None,  # pre-contrast T1 map if using concentration
        TR=None, TE=None, FA_deg=None  # sequence params for spoiled GRE approx (optional)

    Returns:
        washin_map, ttp_map, peak_amp
    """
    print(f'img4d {img4d.shape}')
    if times is None:
        times = img4d.timeline
    result_shape = (1,) if img4d.ndim < 2 else img4d.shape[1:]
    print(f'result_shape {result_shape}')
    # If metric input is a string, compute percent-change same as before
    raw_mode = isinstance(metric, str)
    if raw_mode and metric not in ['percent', 'concentration']:
        raise ValueError("Only 'percent' and 'concentration' supported when passing raw img4d.")
    if raw_mode:
        # compute percent-change metric
        baseline = np.array(np.mean(img4d[:baseline_frames], axis=0))
        if baseline.ndim < 1:
            baseline = np.array(baseline).reshape(1)
            print('Change float into ndarray')
        print(f'baseline {type(baseline)} {baseline.shape}')
        baseline[baseline == 0] = np.finfo(float).eps
        print(f'baseline {type(baseline)} {baseline.shape}')
        print(f'til metric_arr img4d {img4d.shape} baseline[None] {baseline[None].shape}')
        if metric == 'percent':
            metric_arr = 100.0 * (img4d - baseline) / np.abs(baseline)
        elif metric == 'concentration':
            if T10 is None or r1 is None:
                raise ValueError("T10 and r1 required for concentration method.")
            # simple linear approximation: ΔR1 = (1/ T1(t) - 1/T10) ≈ k * (S - S0)/S0
            # Here we implement a commonly used approximate conversion for spoiled GRE:
            if TR is None or FA_deg is None:
                raise ValueError("TR and FA required for spoiled GRE approx.")
            FA = np.deg2rad(FA_deg)
            S0 = np.abs(baseline)
            # approximate proportionality constant k (this is an approximation)
            # k = (1 - E1) / (S0 * sin(FA)) * ... (simplified). For robust conversion prefer T1 mapping.
            # We'll use percent as fallback if robust conversion not desired.
            metric_arr = 100.0 * (img4d - S0) / S0
        if metric_arr.shape[0] == 1:
            metric_arr = metric_arr[0]
    else:
        metric_arr: np.ndarray = metric
    print(f'metric_arr {metric_arr.shape}')

    if smooth_window and smooth_window > 1:
        metric_arr = scipy.ndimage.uniform_filter1d(metric_arr, size=smooth_window, axis=0, mode='nearest')
        # metric_arr = Series(metric_arr, template=img4d, geometry=img4d)

    # nt,nz,ny,nx = metric_arr.shape
    nt = img4d.shape[0]
    ttp_map = np.full(result_shape, np.nan, dtype=float)
    peak_amp = np.zeros(result_shape, dtype=float)
    washin_map = np.full(result_shape, np.nan, dtype=float)
    slope_map = np.full(result_shape, np.nan, dtype=float)
    # ttp_map = np.zeros_like(metric_arr[0], dtype=float); ttp_map[:] = np.nan
    # peak_amp = np.zeros_like(metric_arr[0], dtype=float)
    # washin_map = np.empty_like(metric_arr[0], dtype=float); washin_map[:] = np.nan

    # helper to find arrival and peak (simple reuse of prior logic)
    print(f'metric_arr {metric_arr.shape[1:]}')
    for idx in np.ndindex(metric_arr.shape[1:]):
        curve = metric_arr[tuple((slice(0, nt),)) + idx]
        fig, ax = plt.subplots(1)
        if not raw_mode:
            show(metric[tuple((slice(0, nt),)) + idx], x=times, ax=ax, label='metric', show=False)
        show(curve, x=times, ax=ax, label='curve', show=False)
        if np.all(np.isnan(curve)) or np.all(curve <= 0):
            continue
        baseline_mean = np.mean(curve[:baseline_frames])
        arrival_idx = None
        for t_idx in range(baseline_frames, nt):
            if abs(curve[t_idx]) >= abs(baseline_mean * (1 + arrival_thresh_pct)):
                arrival_idx = max(t_idx - 1, 0)
                break
        if arrival_idx is None:
            continue
        t_arrival = times[arrival_idx]

        # find peak index after arrival
        post_curve = curve[arrival_idx:]
        if post_curve.size == 0:
            continue
        peak_local_idx = np.nanargmax(post_curve)
        peak_frame = arrival_idx + int(peak_local_idx)
        peak_amp[idx] = curve[peak_frame]
        result = first_major_peak_ttp(times, curve,
                                      post_idx_start=arrival_idx,
                                      min_prominence=0.3*float(peak_amp[idx][0]),
                                      min_distance_seconds=15)
        t_shoulder1, x_shoulder1, _idx1 = find_shoulder(times[arrival_idx:],
                                                        curve[arrival_idx:],
                                                        positive_only=True)
        _idx = arrival_idx + _idx1
        peak_frame = result['peak_index']
        if peak_frame is None:
            continue
        peak_frame = min(peak_frame, _idx)
        t_peak = times[peak_frame]
        peak_amp[idx] = curve[peak_frame]
        ttp_map[idx] = (t_peak - (t_injection if t_injection is not None else t_arrival))
        plt.axvline(x=t_arrival); plt.axvline(x=t_peak)

        if method == 'arrival-to-peak':
            dt = t_peak - t_arrival
            if dt <= 0:
                washin_map[idx] = np.nan
            else:
                washin_map[idx] = (curve[peak_frame] - curve[arrival_idx]) / dt
        elif method == 'early':
            # fit linear slope over first early_window_s seconds after arrival
            # gather frames within arrival -> arrival + early_window_s
            end_t = t_arrival + early_window_s
            idxs = np.where((times >= t_arrival) & (times <= end_t))[0]
            if len(idxs) < 2:
                washin_map[idx] = np.nan
            else:
                y = curve[idxs]
                x = times[idxs]
                # robust linear fit (ordinary least squares)
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
                washin_map[idx] = slope  # units metric units per second
                plt.plot((x[0], x[-1]), (slope*x[0]+intercept, slope*x[-1]+intercept))
                plt.plot((x[0], x[-1]), (y[0], y[-1]))  # , ax=ax)
        elif method == 'logistic':
            # Source - https://stackoverflow.com/a/60210529
            # Posted by Brenlla
            # Retrieved 2026-06-18, License - CC BY-SA 4.0

            y = curve[:peak_frame]
            t = times[:peak_frame]

            # t: array of acquisition times (s or min); y: percent-enhancement
            p0 = [np.median(y[:baseline_frames]), y.max(), 1.0, t[np.argmax(y)]]
            bounds = ([-np.inf, 0, 0, t.min()], [np.inf, np.inf, np.inf, t.max()])

            params, cov = curve_fit(logistic, t, y, p0=p0, bounds=bounds)

            A, K, B, M = params

            # Option 1: inflection-based TTP
            TTP_inflection = M

            # Option 2: 95% of K
            p = 0.95
            TTP_p95 = M - (1.0 / B) * np.log((1.0 / p) - 1.0)

            # Option 3: numeric max on fine grid
            t_dense = np.linspace(t.min(), t.max(), 1000)
            y_dense = logistic(t_dense, *params)
            TTP_numeric = t_dense[np.argmax(y_dense)]

            print(f'TTP_inflection: {TTP_inflection}, TTP_p95: {TTP_p95}, TTP_numeric: {TTP_numeric}')
            ttp_map[idx] = TTP_p95
            # peak_amp[idx] = K
            washin_map[idx] = B
            try:
                slope_map[idx] = np.max(np.diff(y[baseline_frames:peak_frame]))
            except ValueError:
                pass

            if savefig is not None or vis:
                fig, ax = plt.subplots(1, sharex=True)
                ax.scatter(times, curve, label='Tissue')
                ax.plot(t, logistic(t, *params), 'r-', label='Fitted function')
                print('params:', params)
                print('cov:', cov)
                # ax.plot(x, dydx, 'y-', label='Slope')
                ax.axvline(x=t_arrival, color='y', linestyle='-', label='Arrival')
                ax.axvline(x=t_peak, color='y', linestyle='--')
                # ax.axvline(x=M-TTP_numeric/2, color='b', linestyle='--', label='TTP_numeric')
                # ax.axvline(x=M+TTP_numeric/2, color='b', linestyle='--')
                ax.axvline(x=M-TTP_p95/2, color='k', linestyle='--', label='TTP_p95')
                ax.axvline(x=M+TTP_p95/2, color='k', linestyle='--')
                ax.axvline(x=M-TTP_inflection/2, color='g', linestyle='--', label='TTP_inflection')
                ax.axvline(x=M+TTP_inflection/2, color='g', linestyle='--')
                ax.legend()
                if savefig is not None:
                    plt.savefig(savefig)

        else:
            raise ValueError("method must be 'early', 'arrival-to-peak', 'logistic'")
    plt.show()

    if issubclass(type(img4d), Series):
        _ttp_map = Series(np.nan_to_num(ttp_map, posinf=0, neginf=0), template=img4d, geometry=img4d)
        _ttp_map.seriesDescription = 'TTP'; _ttp_map.window=300; _ttp_map.level=150
        _peak_amp = Series(np.nan_to_num(peak_amp, posinf=0, neginf=0), template=img4d, geometry=img4d)
        _peak_amp.seriesDescription = 'Peak amplitude'; _peak_amp.window=30; _peak_amp.level=15
        _washin_map = Series(np.nan_to_num(washin_map, posinf=0, neginf=0), template=img4d, geometry=img4d)
        _washin_map.seriesDescription = 'Wash-in slope [1/s]'; _washin_map.window = 60; _washin_map.level = 0
        _slope_map = Series(np.nan_to_num(slope_map, posinf=0, neginf=0), template=img4d, geometry=img4d)
        _slope_map.seriesDescription = 'Wash-in slope [1/s]'; _slope_map.window = 60; _slope_map.level = 0
        return _washin_map, _slope_map, _ttp_map, _peak_amp
    else:
        return washin_map, slope_map, ttp_map, peak_amp


def compute_ttp(
    img4d,              # numpy array shape (nx,ny,nz,nt)
    times,              # 1D array length nt, acquisition time (s) of each frame
    method='percent',   # 'percent' or 'concentration'
    baseline_frames=5,  # number of initial frames to compute baseline
    smooth_window=3,    # temporal smoothing window (frames), set 0 to disable
    arrival_thresh_pct=5.0,  # % signal change threshold to detect bolus arrival
    t_injection=None,   # optional injection time in seconds (if you want TTP from injection)
    r1=None,            # relaxivity (s^-1 mM^-1) if using concentration conversion
    T10=None,           # pre-contrast T1 map shape (nx,ny,nz) if using concentration
    TR=None, TE=None, FA_deg=None  # sequence params for spoiled GRE approx (optional)
):
    nx,ny,nz,nt = img4d.shape
    assert len(times) == nt

    # baseline (mean over baseline_frames)
    baseline = np.mean(img4d[..., :baseline_frames], axis=3)
    baseline[baseline == 0] = np.finfo(float).eps

    # compute signal-change metric per voxel
    if method == 'percent':
        # percent signal change relative to baseline
        pct = 100.0 * (img4d - baseline[..., None]) / baseline[..., None]
        metric = pct
    elif method == 'concentration':
        if T10 is None or r1 is None:
            raise ValueError("T10 and r1 required for concentration method.")
        # simple linear approximation: ΔR1 = (1/ T1(t) - 1/T10) ≈ k * (S - S0)/S0
        # Here we implement a commonly used approximate conversion for spoiled GRE:
        if TR is None or FA_deg is None:
            raise ValueError("TR and FA required for spoiled GRE approx.")
        FA = np.deg2rad(FA_deg)
        S0 = baseline
        # approximate proportionality constant k (this is an approximation)
        # k = (1 - E1) / (S0 * sin(FA)) * ... (simplified). For robust conversion prefer T1 mapping.
        # We'll use percent as fallback if robust conversion not desired.
        metric = 100.0 * (img4d - S0[..., None]) / S0[..., None]
    else:
        raise ValueError("method must be 'percent' or 'concentration'")

    # optional temporal smoothing
    if smooth_window and smooth_window > 1:
        metric = scipy.ndimage.uniform_filter1d(metric, size=smooth_window, axis=3, mode='nearest')

    # outputs
    ttp_map = np.full((nx,ny,nz), np.nan, dtype=float)
    peak_amp = np.zeros((nx,ny,nz), dtype=float)

    frame_dt = np.diff(times)
    # compute for each voxel
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                curve = metric[ix,iy,iz,:]
                # skip flat or zero baseline voxels
                if np.all(np.isnan(curve)) or np.all(curve <= 0):
                    continue
                # bolus arrival time: first frame after baseline where metric > threshold
                baseline_mean = np.mean(curve[:baseline_frames])
                baseline_std = np.std(curve[:baseline_frames]) if baseline_frames>1 else 0.0
                arrival_idx = None
                for t_idx in range(baseline_frames, nt):
                    if curve[t_idx] >= baseline_mean + arrival_thresh_pct:
                        arrival_idx = t_idx
                        break
                if arrival_idx is None:
                    # no clear arrival: mark undefined
                    continue

                # find peak after arrival
                post_curve = curve[arrival_idx:]
                if post_curve.size == 0:
                    continue
                peak_local_idx = np.nanargmax(post_curve)
                peak_frame = arrival_idx + int(peak_local_idx)
                peak_amp[ix,iy,iz] = curve[peak_frame]

                # sub-frame peak interpolation using cubic spline if possible and not at boundaries
                if nt >= 4:
                    tt = times[arrival_idx:arrival_idx+len(post_curve)]
                    try:
                        cs = scipy.interpolate.CubicSpline(tt, post_curve, bc_type='natural')
                        dense_t = np.linspace(tt[0], tt[-1], max(100, len(tt)*10))
                        dense_vals = cs(dense_t)
                        dense_peak_idx = np.nanargmax(dense_vals)
                        t_peak = dense_t[dense_peak_idx]
                    except Exception:
                        t_peak = times[peak_frame]
                else:
                    t_peak = times[peak_frame]

                if t_injection is not None:
                    ttp = t_peak - t_injection
                else:
                    # default TTP relative to bolus arrival time
                    t_bolus = times[arrival_idx]
                    ttp = t_peak - t_bolus

                ttp_map[ix,iy,iz] = ttp

    return ttp_map, peak_amp


def compute_auc_maps(
    img4d, times,
    baseline_frames=5,
    smooth_window=3,
    arrival_thresh_pct=5.0,
    windows_s=(30.0, 60.0),   # compute AUCs for these durations after arrival
    method='percent'          # 'percent' or precomputed metric ndarray
):
    # prepare metric array
    result_shape = (1,) if img4d.ndim < 2 else img4d.shape[1:]
    raw_mode = isinstance(method, str) and method == 'percent'
    if raw_mode:
        baseline = np.mean(img4d[:baseline_frames], axis=0)
        baseline[baseline == 0] = np.finfo(float).eps
        metric = 100.0 * (img4d - baseline) / baseline
    else:
        metric = method  # user supplied metric array

    if smooth_window and smooth_window > 1:
        metric = scipy.ndimage.uniform_filter1d(metric, size=smooth_window, axis=0, mode='nearest')

    nt = img4d.shape[0]
    auc_total = np.full(result_shape, np.nan, dtype=float)        # integral over full timeseries
    auc_from_arrival = np.full(result_shape, np.nan, dtype=float) # integral from detected arrival to end
    auc_fixed = {w: np.full(result_shape, np.nan, dtype=float) for w in windows_s}
    auc_fixed_norm = {w: np.full(result_shape, np.nan, dtype=float) for w in windows_s}

    for idx in np.ndindex(metric.shape[1:]):
        curve = metric[tuple((slice(0, nt),)) + idx]
        if np.all(np.isnan(curve)):
            continue
        # total AUC using trapezoid
        auc_total[idx] = np.trapezoid(curve, times)

        # detect arrival
        baseline_mean = np.mean(curve[:baseline_frames])
        arrival_idx = None
        for t_idx in range(baseline_frames, nt):
            if curve[t_idx] >= baseline_mean + arrival_thresh_pct:
                arrival_idx = t_idx
                break
        if arrival_idx is None:
            continue
        t_arrival = times[arrival_idx]

        # AUC from arrival to end
        auc_from_arrival[idx] = np.trapezoid(curve[arrival_idx:], times[arrival_idx:])

        # fixed-window AUCs after arrival (interpolate if window extends between frames)
        for w in windows_s:
            t_end = t_arrival + w
            # if t_end beyond last time, integrate to last time and note shorter duration
            if t_end > times[-1]:
                # integrate available range
                auc_val = np.trapezoid(curve[arrival_idx:], times[arrival_idx:])
                actual_dur = times[-1] - t_arrival
            else:
                # find indices covering [t_arrival, t_end]; include interpolation at t_end
                # find index of first frame >= t_end
                idx_end = np.searchsorted(times, t_end)
                # gather times and values from arrival_idx to idx_end (inclusive)
                t_seg = times[arrival_idx:idx_end+1]
                y_seg = curve[arrival_idx:idx_end+1]
                # if t_end not exactly a sample, interpolate value at t_end
                if t_seg[-1] > t_end:
                    # should not happen due to searchsorted logic, but handle generally
                    pass
                if t_seg[-1] < t_end:
                    y_end = np.interp(t_end, times, curve)
                    t_seg = np.append(t_seg, t_end)
                    y_seg = np.append(y_seg, y_end)
                auc_val = np.trapezoid(y_seg, t_seg)
                actual_dur = t_seg[-1] - t_arrival  # w
            auc_fixed[w][idx] = auc_val
            auc_fixed_norm[w][idx] = auc_val / max(actual_dur, 1e-8)

    return {
        'auc_total': auc_total,
        'auc_from_arrival': auc_from_arrival,
        'auc_fixed': auc_fixed,
        'auc_fixed_norm': auc_fixed_norm
    }


def find_shoulder(t, x, window_frac=0.05, polyorder=3, min_dist_frac=0.02,
                  positive_only=False):
    """
    Returns t_shoulder, x_shoulder, idx

    t: 1D array (monotonic or evenly sampled is easiest)
    x: 1D signal
    """
    t = np.asarray(t)
    x = np.asarray(x)
    n = len(x)

    if n < 7:
        raise ValueError("Need more points to reliably smooth/estimate curvature.")

    # If t is non-uniform, resample would be better; here we assume near-uniform.
    # Derivative scale uses median dt.
    dt = np.median(np.diff(t))

    # Choose Savitzky–Golay window (odd, >= polyorder+2)
    win = int(max(7, np.round(window_frac * n)))
    if win % 2 == 0:
        win += 1
    win = max(win, polyorder + 2 + (polyorder + 2) % 2)  # keep odd-ish safeguard

    # Smooth signal
    xs = savgol_filter(x, window_length=win, polyorder=polyorder, mode="interp")

    # First and second derivatives
    dx = np.gradient(xs, dt)
    d2x = np.gradient(dx, dt)

    # Curvature measure: magnitude of second derivative
    if positive_only:
        curv = d2x
    else:
        curv = np.abs(d2x)

    # Avoid edges where derivatives are noisier
    edge = win // 2 + 2
    curv2 = curv.copy()
    curv2[:edge] = 0
    curv2[-edge:] = 0

    # Pick the most prominent curvature peak (you can tune prominence)
    min_dist = int(max(1, np.round(min_dist_frac * n)))
    peaks, props = find_peaks(curv2, distance=min_dist)

    if len(peaks) == 0:
        idx = int(np.argmax(curv2))
    else:
        # strongest curvature peak
        idx = peaks[np.argmax(curv2[peaks])]

    return t[idx], x[idx], idx

# --- example ---
# t = np.linspace(0, 10, 2000)
# x = ... your signal ...
# t_sh, x_sh, idx = find_shoulder(t, x)
# print(t_sh, x_sh, idx)
