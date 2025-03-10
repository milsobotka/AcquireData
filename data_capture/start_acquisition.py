import subprocess

camera_script = r"C:/Users/admin/Desktop/phd-workspace/AcquireData/VmbPy-main/Examples/main_u130vswir.py"
#camera_script = r"C:/Users/admin/Desktop/phd-workspace/AcquireData/logitech_camera/logitech_camera.py"
pulse_data_script = r"C:/Users/admin/Desktop/phd-workspace/AcquireData/hl7_parser/hl7_data_collector.py"

process_camera = subprocess.Popen(["python", camera_script])
process_pulse = subprocess.Popen(["python", pulse_data_script])

process_camera.wait()
process_pulse.wait()

print("Data collection finished successfully")
