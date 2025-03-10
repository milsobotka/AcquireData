import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import butter, filtfilt
import cv2

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_snr1(fft_values, freq, band=(0.7, 4.0)):
    band_mask = (freq >= band[0]) & (freq <= band[1])
    signal_power = np.sum(fft_values[band_mask] ** 2)
    noise_power = np.sum(fft_values[~band_mask] ** 2)
    if noise_power <= 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)

def get_snr2(signal, fs, lp_freq=0.7, hp_freq=4.0):
    N = len(signal)
    delta_f = fs / N
    f1 = int(lp_freq / delta_f)
    f2 = int(hp_freq / delta_f)
    ps = np.abs(np.fft.fft(signal))**2
    ps_pos = ps[:N//2]
    snr_max = -1000.0
    for i in range(f1, f2):
        sum_v = np.sum(ps_pos[i:i+10])
        sum_r = np.sum(ps_pos[:i]) + np.sum(ps_pos[i+10:])
        snr = sum_v / sum_r if sum_r > 0 else sum_v
        if snr > snr_max:
            snr_max = snr
    return 10.0 * np.log10(snr_max) if snr_max > 0 else 0

def compute_fft(signal, fs):
    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals)[:len(signal)//2]
    freq = np.fft.fftfreq(len(signal), d=1/fs)[:len(signal)//2]
    return fft_vals, freq

def detect_face(frame):
    haarcascade_path = "C:/Users/admin/Desktop/phd-workspace/AcquireData/signal_processing/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return (x, y, w, h)
    else:
        return None

def extract_roi(frames, center_size=(200, 200)):
    num_frames, height, width, _ = frames.shape
    first_frame = frames[0].astype(np.uint8)
    face_rect = detect_face(first_frame)

    if face_rect is not None:
        x, y, w, h = face_rect
        x2 = min(x + w, width)
        y2 = min(y + h, height)
        cropped = []
        for i in range(num_frames):
            frame_bgr = frames[i].astype(np.uint8)
            roi = frame_bgr[y:y2, x:x2, :]
            cropped.append(roi)
        return np.array(cropped), face_rect, True
    else:
        cy, cx = height // 2, width // 2
        hh, hw = center_size[0] // 2, center_size[1] // 2
        cropped = frames[:, cy - hh : cy + hh, cx - hw : cx + hw, :]
        return cropped, None, False

def visualize_frame_roi(frame_bgr, output_path, face_rect=None, center_size=(200, 200)):
    frame_rgb = frame_bgr[..., ::-1].astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(frame_rgb)

    if face_rect is not None:
        x, y, w, h = face_rect
        rect = plt.Rectangle((x, y), w, h, edgecolor='red', linewidth=2, facecolor='none')
        ax.add_patch(rect)
        plt.title("Detected face (ROI)")
    else:
        h, w, _ = frame_rgb.shape
        cy, cx = h // 2, w // 2
        hh, hw = center_size[0] // 2, center_size[1] // 2
        rect = plt.Rectangle((cx - hw, cy - hh), center_size[1], center_size[0],
                             edgecolor='red', linewidth=2, facecolor='none')
        ax.add_patch(rect)
        plt.title("Center ROI 200x200")

    plt.savefig(output_path)
    plt.close()

def process_video_file(file_path, base_output_dir, input_directory):
    relative_path = os.path.relpath(os.path.dirname(file_path), start=input_directory)
    output_dir = os.path.join(base_output_dir, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    video_frames = np.load(file_path)
    num_frames, height, width, channels = video_frames.shape
    fps = num_frames / 30.0

    frames_roi, face_rect, face_found = extract_roi(video_frames)
    visualize_frame_roi(
        video_frames[0],
        os.path.join(output_dir, "sample_frame_with_roi.png"),
        face_rect=face_rect
    )

    r = frames_roi[..., 0].mean(axis=(1, 2))
    g = frames_roi[..., 1].mean(axis=(1, 2))
    b = frames_roi[..., 2].mean(axis=(1, 2))
    avg = (r + g + b) / 3.0

    fr = butter_bandpass_filter(r, 0.7, 4.0, fps)
    fg = butter_bandpass_filter(g, 0.7, 4.0, fps)
    fb = butter_bandpass_filter(b, 0.7, 4.0, fps)
    fall = butter_bandpass_filter(avg, 0.7, 4.0, fps)

    np.save(os.path.join(output_dir, "filtered_signals.npy"), np.array([fr, fg, fb, fall]))

    rr_fft, freq_r   = compute_fft(r, fps)
    rg_fft, _        = compute_fft(g, fps)
    rb_fft, _        = compute_fft(b, fps)
    rall_fft, _      = compute_fft(avg, fps)
    fr_fft, freq_fr  = compute_fft(fr, fps)
    fg_fft, _        = compute_fft(fg, fps)
    fb_fft, _        = compute_fft(fb, fps)
    fall_fft, _      = compute_fft(fall, fps)

    snr1_r   = get_snr1(rr_fft, freq_r)
    snr1_g   = get_snr1(rg_fft, freq_r)
    snr1_b   = get_snr1(rb_fft, freq_r)
    snr1_all = get_snr1(rall_fft, freq_r)

    snr2_r   = get_snr2(fr, fps)
    snr2_g   = get_snr2(fg, fps)
    snr2_b   = get_snr2(fb, fps)
    snr2_all = get_snr2(fall, fps)

    frame_index = np.arange(num_frames)
    df_time = pd.DataFrame({
        "Frame": frame_index,
        "Brightness_R": r,
        "Brightness_G": g,
        "Brightness_B": b,
        "Brightness_Avg": avg,
        "Filtered_R": fr,
        "Filtered_G": fg,
        "Filtered_B": fb,
        "Filtered_Avg": fall
    })
    df_time.to_csv(os.path.join(output_dir, "time_data.csv"), index=False)

    df_fft = pd.DataFrame({
        "Freq_R": freq_r,
        "RawFFT_R": rr_fft,
        "FiltFFT_R": fr_fft,
        "RawFFT_G": rg_fft,
        "FiltFFT_G": fg_fft,
        "RawFFT_B": rb_fft,
        "FiltFFT_B": fb_fft,
        "RawFFT_Avg": rall_fft,
        "FiltFFT_Avg": fall_fft
    })
    df_fft.to_csv(os.path.join(output_dir, "fft_data.csv"), index=False)

    snr_csv = os.path.join(output_dir, "snr_data.csv")
    df_snr = pd.DataFrame([{
        "File": file_path,
        "SNR1_R": snr1_r,   "SNR1_G": snr1_g,   "SNR1_B": snr1_b,   "SNR1_Avg": snr1_all,
        "SNR2_R": snr2_r,   "SNR2_G": snr2_g,   "SNR2_B": snr2_b,   "SNR2_Avg": snr2_all
    }])
    if not os.path.exists(snr_csv):
        df_snr.to_csv(snr_csv, index=False, mode='w')
    else:
        df_snr.to_csv(snr_csv, index=False, mode='a', header=False)

    channels = [
        ("Red", fr, fr_fft, freq_fr, "red"),
        ("Green", fg, fg_fft, freq_fr, "green"),
        ("Blue", fb, fb_fft, freq_fr, "blue"),
        ("Avg", fall, fall_fft, freq_fr, "black"),
    ]
    for name, fsig, fsig_fft, fsig_freq, color in channels:
        plt.figure()
        plt.plot(frame_index, fsig, color=color, label=f"{name} - FILTERED")
        plt.xlabel("Frame")
        plt.ylabel("Brightness")
        plt.title(f"{name} Filtered Signal")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{name}_filtered_signal.png"))
        plt.close()

        plt.figure()
        plt.plot(fsig_freq, fsig_fft, color=color, linestyle='dashed', label=f"{name} - FILTERED FFT")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.title(f"{name} Filtered FFT")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{name}_filtered_fft.png"))
        plt.close()

    print(f"\n[INFO] Processed file: {file_path}")
    if face_found:
        print("  -> Face detected. Using face region.")
    else:
        print("  -> No face detected. Using center 200x200 region.")
    print(f"  -> Results in: {output_dir}\n")

if __name__ == "__main__":
    base_output_dir = "C:/Users/admin/Desktop/phd-workspace/output_data/logitech_carl_zeiss/2025-03-10_15-47-44.275984"
    input_directory = "C:/Users/admin/Desktop/phd-workspace/input_data/logitech_carl_zeiss/2025-03-10_15-47-44.275984"


    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".npy") and not file.endswith(".json.npy"):
                file_path = os.path.join(root, file)
                process_video_file(file_path, base_output_dir, input_directory)

    print("\nProcessing of all .npy files finished.\n")
