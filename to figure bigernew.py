import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import numpy as np

# Set the directory containing audio files
folder_path = r"C:\Users\ACER\Desktop\meaw\dtafa\test\CS"

# Change directory to the folder containing audio files
os.chdir(folder_path)

# Get a list of all audio files in the folder
audio_files = [file for file in os.listdir() if file.endswith('.wav')]

# Loop through each audio file in the folder
for file in audio_files:
    # Load audio file
    y, sr = librosa.load(file)

    # Compute spectrogram
    n_fft = 2048
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)

    # Display spectrogram
    plt.figure(figsize=(8, 6))  # Reduce the figure size
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of {}'.format(file))
    plt.ylim(512, None)  # Set y-axis limits starting from 256
    
    # Adjust DPI to control image resolution
    dpi = 300  # You can adjust this value as needed
    plt.savefig(os.path.join(folder_path, file.replace('.wav', '_s.png')), dpi=dpi)
    plt.close()
