import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(arr):
    arr = arr - np.min(arr)
    mx = np.max(arr)
    return arr / mx if mx > 0 else arr

def get_resampled_signals(hl7_path, cam_path, fps_cam, t_min=0.0, t_max=30.0):
    df_hl7 = pd.read_csv(hl7_path, sep=';')
    sig_hl7 = df_hl7["pulse_value"].values
    N_hl7   = len(sig_hl7)
    fs_hl7 = N_hl7 / (t_max - t_min)
    time_hl7 = np.arange(N_hl7) / fs_hl7

    df_cam = pd.read_csv(cam_path)
    frames = df_cam["Frame"].values
    sig_cam = df_cam["Filtered_G"].values
    time_cam = frames / fps_cam

    num_samples = max(len(sig_hl7), len(sig_cam))
    time_common = np.linspace(t_min, t_max, num_samples)

    sig_hl7_resampled = np.interp(time_common, time_hl7, sig_hl7)
    sig_hl7_resampled = normalize(sig_hl7_resampled)

    sig_cam_resampled = np.interp(time_common, time_cam, sig_cam)
    sig_cam_resampled = normalize(sig_cam_resampled)

    return time_common, sig_hl7_resampled, sig_cam_resampled

#
hl7_path_u235c = "C:/Users/admin/Desktop/phd-workspace/HL7_processed_data/u235c/2025-03-10_15-11-08.495322_pulse.csv"
cam_path_u235c = "C:/Users/admin/Desktop/phd-workspace/output_data/u235c/2025-03-10_15-11-08.585819/time_data.csv"
fps_u235c    = 88.77398492297864 

time_1, hl7_1, cam_1 = get_resampled_signals(hl7_path_u235c, cam_path_u235c, fps_u235c)

plt.figure(figsize=(15,7))
plt.plot(time_1, hl7_1, label="HL7 (u235c)")
plt.plot(time_1, cam_1,  label="Kamera (u235c)")
plt.title("Kamera u235c")
plt.xlabel("Czas [s]")
plt.ylabel("Znormalizowana amplituda")
plt.legend()
plt.grid(True)
plt.savefig("Kamera u235c")
plt.show()

#
hl7_path_u130 = "C:/Users/admin/Desktop/phd-workspace/HL7_processed_data/u130vswir/2025-03-10_16-04-14.097883_pulse.csv"
cam_path_u130 = "C:/Users/admin/Desktop/phd-workspace/output_data/u130vswir/2025-03-10_16-04-14.166994/time_data.csv"
fps_u130      = 64.7826044781998

time_2, hl7_2, cam_2 = get_resampled_signals(hl7_path_u130, cam_path_u130, fps_u130)

plt.figure(figsize=(15,7))
plt.plot(time_2, hl7_2, label="HL7 (u130vswir)")
plt.plot(time_2, cam_2,  label="Kamera (u130vswir)")
plt.title("Kamera u130vswir")
plt.xlabel("Czas [s]")
plt.ylabel("Znormalizowana amplituda")
plt.legend()
plt.grid(True)
plt.savefig("Kamera u130vswir")
plt.show()

#
hl7_path_u511 = "C:/Users/admin/Desktop/phd-workspace/HL7_processed_data/u511c/2025-03-10_15-55-19.778709_pulse.csv"
cam_path_u511 = "C:/Users/admin/Desktop/phd-workspace/output_data/u511c/2025-03-10_15-55-19.855117/time_data.csv"
fps_u511      = 44.520946903199025

time_4, hl7_4, cam_4 = get_resampled_signals(hl7_path_u511, cam_path_u511, fps_u511)

plt.figure(figsize=(15,7))
plt.plot(time_4, hl7_4, label="HL7 (u511c)")
plt.plot(time_4, cam_4,  label="Kamera (u511c)")
plt.title("Kamera u511c")
plt.xlabel("Czas [s]")
plt.ylabel("Znormalizowana amplituda")
plt.legend()
plt.grid(True)
plt.savefig("Kamera u511c")
plt.show()

#
hl7_paths_all = [
    hl7_path_u130,
    hl7_path_u235c,
    hl7_path_u511
]
cam_paths_all = [
    cam_path_u130,
    cam_path_u235c,
    cam_path_u511
]
fps_all = [
    fps_u130,
    fps_u235c,
    fps_u511
]

time_arrays = []
hl7_arrays  = []
cam_arrays  = []
lengths_all = []

for hp, cp, f in zip(hl7_paths_all, cam_paths_all, fps_all):
    df_hl7 = pd.read_csv(hp, sep=';')
    sig_hl7_ = df_hl7["pulse_value"].values
    N_hl7_   = len(sig_hl7_)
    fs_hl7_  = N_hl7_ / 30.0
    t_hl7_   = np.arange(N_hl7_) / fs_hl7_

    df_cam_  = pd.read_csv(cp)
    frames_  = df_cam_["Frame"].values
    sig_cam_ = df_cam_["Filtered_G"].values
    t_cam_   = frames_ / f

    lengths_all.append(len(sig_hl7_))
    lengths_all.append(len(sig_cam_))

num_samples_5 = max(lengths_all)

t_min = 0.0
t_max = 30.0
time_common_5 = np.linspace(t_min, t_max, num_samples_5)

hl7_resampled_all = []
cam_resampled_all = []

for hp, cp, f in zip(hl7_paths_all, cam_paths_all, fps_all):
    df_hl7 = pd.read_csv(hp, sep=';')
    sig_hl7_ = df_hl7["pulse_value"].values
    N_hl7_   = len(sig_hl7_)
    fs_hl7_  = N_hl7_ / 30.0
    t_hl7_   = np.arange(N_hl7_) / fs_hl7_

    sig_hl7_interp = np.interp(time_common_5, t_hl7_, sig_hl7_)
    sig_hl7_interp = normalize(sig_hl7_interp)

    df_cam_  = pd.read_csv(cp)
    frames_  = df_cam_["Frame"].values
    sig_cam_ = df_cam_["Filtered_G"].values
    t_cam_   = frames_ / f

    sig_cam_interp = np.interp(time_common_5, t_cam_, sig_cam_)
    sig_cam_interp = normalize(sig_cam_interp)

    hl7_resampled_all.append(sig_hl7_interp)
    cam_resampled_all.append(sig_cam_interp)

plt.figure(figsize=(15,7))

plt.plot(time_common_5, hl7_resampled_all[0], color='deeppink', label="HL7 (u130vswir)")
plt.plot(time_common_5, cam_resampled_all[0], color='lightpink', label="Kamera (u130vswir)")

plt.plot(time_common_5, hl7_resampled_all[1], color='lightseagreen', label="HL7 (u235c)")
plt.plot(time_common_5, cam_resampled_all[1], color='paleturquoise', label="Kamera (u235c)")

plt.plot(time_common_5, hl7_resampled_all[2], color='coral', label="HL7 (u511c)")
plt.plot(time_common_5, cam_resampled_all[2], color='lightcoral', label="Kamera (u511c)")

plt.xlabel("Czas [s]")
plt.ylabel("Znormalizowana amplituda")
plt.legend()
plt.grid(True)
plt.savefig("Kamera u130vswir, u235c, u511c")
plt.show()

