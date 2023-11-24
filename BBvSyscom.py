#Dependencies: python/BBVerificationTest.py; pip install librosa pandas openpyxl; 
#Note this is m/s not mm/s *******************************
#------------------ Importing Libraries + Setup ------------------#
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Set this variable to be the peak number that you would like to see the
# spectral copmparison for
pk_number = 1; # Picking the number peak for which the detailed comparison
            # will be performed run the code once and adjust as per required.

frequencyLimits = [1, 100]; # Hz
# Syscom Sampling rate (Hz)
syscom_SR = 2000
# Big Bertha Sampling rate (Hz)
BB_SR = 800

#------------------ Big Bertha Data ------------------#
# Big Bertha data is stored in the Time Table data structure
# read the wav file to get velocity and time for BB (mm/s)


def calculate_velocity(audio_file_path, window_size=0.01):
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_file_path)

    # Calculate the velocity (mm/s) from audio data
    velocity = librosa.feature.rms(y=audio_data, frame_length=int(sample_rate*window_size))
    velocity = velocity * 1000
    # Calculate time values for each velocity point
    time = librosa.times_like(velocity)

    return time, velocity[0]

def create_timetable_wav(time, velocity):
    # Create a DataFrame (timetable) using pandas
    timetable = pd.DataFrame({'Time (s)': time, 'Velocity (m/s)': velocity})

    return timetable

if __name__ == "__main__":
    # Replace with the path to your .wav file
    audio_file_path = r"C:\Users\boussenane\Desktop\C\python\BB-data.wav"

    time, velocity = calculate_velocity(audio_file_path)
    BB_timetable = create_timetable_wav(time, velocity)
    print("Big Bertha data")
    print(BB_timetable)
    print("great success!!!")
#------------------ Big Bertha Data ------------------#

#------------------ Syscom Data ------------------#
def create_timetable_excel(data):
    # Create a DataFrame (timetable) using pandas
    timetable = pd.DataFrame({
        'Time': data.iloc[:, 0],  # Assumes column 1 contains time data
        'Velocity': data.iloc[:, 3]  # Assumes column 4 contains velocity data
    })

    return timetable

if __name__ == "__main__":
    excel_file_path = r"C:\Users\boussenane\Desktop\C\python\SyscomData.xlsx"  # Replace with the path to your Excel file

    # Read the Excel file
    data = pd.read_excel(excel_file_path)

    # Create the timetable
    syscom_timetable = create_timetable_excel(data)
    print("Syscom data")
    print(syscom_timetable)
    print("great success!!!")
#------------------ Syscom Data ------------------#

#------------------ Find and Plot Syscom Peaks ------------------#

# Find peaks in syscom_timetable
peaks, _ = find_peaks(syscom_timetable['Velocity'], distance=syscom_SR/2, prominence=0.5)

# Plot the original data
plt.plot(syscom_timetable['Time'], syscom_timetable['Velocity'])

# Mark the peaks on the plot
plt.plot(syscom_timetable['Time'].iloc[peaks], syscom_timetable['Velocity'].iloc[peaks], "x", label="Peaks")

# Annotate the peaks with their index
for i, (time, velocity) in enumerate(zip(syscom_timetable['Time'].iloc[peaks], syscom_timetable['Velocity'].iloc[peaks])):
    plt.text(time + 0.02, velocity, str(i + 1))

plt.xlabel("Time")
plt.ylabel("Velocity")
plt.legend(["Velocity", "Peaks"])
plt.grid(True)

# Save the plot as an image
plt.savefig("SyscomPeaks.png")

# Show the plot (optional)
plt.show()
#------------------ Find and Plot Peaks ------------------#

#------------------ Remove time delay ------------------#

#how to calculate the time delay between the two systems?

# Insert a manual time difference to remove the time delay
BB_start_time = 4.69
sys_start_time = 3.64875
BB_timetable['Time'] = BB_timetable['Time'] - (BB_start_time - sys_start_time)
#------------------ Remove time delay ------------------#


#------------------ Comparison ------------------#
"""
# Find peaks in Syscom data
peaks, _ = find_peaks(syscom_timetable['Velocity'], distance=syscom_SR/2, prominence=0.5)

# Loop through each peak
for pk_number in range(1, len(peaks) + 1):
    peak_index = peaks[pk_number - 1]
    time_window = [max(peak_index - 1, 0), peak_index + 1]

    # Plot original data
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(syscom_timetable['Time'], syscom_timetable['Velocity'], label='Syscom')
    plt.plot(BB_timetable['Time'], BB_timetable['Velocity'], label='Big Bertha')
    plt.legend()

    plt.subplot(2, 1, 2)

    # Calculate and plot power spectra
    f_syscom, Pxx_syscom = welch(syscom_timetable['Velocity'][time_window[0]:time_window[1]], fs=FsSyscom)
    f_BB, Pxx_BB = welch(BB_timetable['Velocity'][time_window[0]:time_window[1]], fs=FsBB)
    plt.semilogy(f_syscom, Pxx_syscom, label='Syscom')
    plt.semilogy(f_BB, Pxx_BB, label='Big Bertha')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Peak_{pk_number}.png')
    plt.close()

    # Octave-band filtering
    f_center = frequencyLimits[1]

    while f_center > frequencyLimits[0]:
        octave_band = [f_center / np.sqrt(2), f_center * np.sqrt(2)]

        # Apply bandpass filtering
        syscom_filtered = bandpass(syscom_timetable['Velocity'][time_window[0]:time_window[1]], octave_band, FsSyscom)
        BB_filtered = bandpass(BB_timetable['Velocity'][time_window[0]:time_window[1]], octave_band, FsBB)

        # Plot filtered data
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(syscom_timetable['Time'][time_window[0]:time_window[1]], syscom_filtered, label='Syscom')
        plt.plot(BB_timetable['Time'][time_window[0]:time_window[1]], BB_filtered, label='Big Bertha')
        plt.legend()
        plt.title(f'{f_center} Hz Octave Band')

        plt.tight_layout()
        plt.savefig(f'OctaveBandForPeak_{pk_number}_{f_center}Hz.png')
        plt.close()

        f_center /= 2
"""
#------------------ Comparison ------------------#