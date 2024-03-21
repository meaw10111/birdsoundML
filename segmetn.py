import os
import librosa
from pydub import AudioSegment
import soundfile as sf

def split_audio(input_file, output_folder, segment_duration, ffmpeg_path):
    # Set ffmpeg executable path
    AudioSegment.ffmpeg = ffmpeg_path

    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Calculate the number of segments
    num_segments = len(audio) // (segment_duration * 1000)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each segment and save it to the output folder
    for i in range(num_segments):
        start_time = i * segment_duration * 1000  # Convert to milliseconds
        end_time = (i + 1) * segment_duration * 1000  # Convert to milliseconds
        segment = audio[start_time:end_time]
        output_file = os.path.join(output_folder, f"segment_{i+1}.wav")
        segment.export(output_file, format='wav')
        

input_file = r"C:\Users\ACER\Desktop\meaw\Eudynamys_scolopaceus_T\900104.wav"  # Replace with the path to your input audio file
output_folder = "output"  # Replace with the folder where you want to save the segments
segment_duration = 5  # Duration of each segment in seconds
ffmpeg_path = r"C:\path\ffmpeg.exe"

split_audio(input_file, output_folder, segment_duration, ffmpeg_path)
