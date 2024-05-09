import torch
import numpy as np
import pyaudio
import IPython.display as ipd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
import wave

# Load the model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

# Function to record audio from microphone
def record_audio(seconds=5, sr=16000):
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = sr
    RECORD_SECONDS = seconds

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=chunk)

    print("* Recording audio...")

    frames = []

    for i in range(0, int(RATE / chunk * RECORD_SECONDS)):
        data = stream.read(chunk)
        frames.append(data)

    print("* Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames), RATE

def start_audio_recording(output_file="audio.wav", duration=5, sample_rate=44100, channels=2, format=pyaudio.paInt16):
    chunk = 1024  # Record in chunks
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Recording audio...")

    frames = []

    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio to a file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    return output_file

def stop_audio_recording():
    return record_audio(duration=5, sr=16000)


# Function to convert audio bytes to tensor
def audio_to_input_values(audio):
    audio_array = np.frombuffer(audio, dtype=np.int16)
    tensor = torch.tensor(audio_array).float()
    return tensor

# Function to transcribe audio
def transcribe_audio(audio):
    if audio is None:
        return ""  # Return empty string if audio is None

    input_values = audio_to_input_values(audio)
    with torch.no_grad():
        logits = model(input_values.unsqueeze(0)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

# Function to detect emotion from transcription using sentiment analysis and keyword matching
def detect_emotion(transcription):
    if not transcription:
        return "neutral"  # Return neutral if transcription is empty

    # Analyze sentiment of the transcription
    sentiment_result = sentiment_analysis(transcription)
    sentiment_label = sentiment_result[0]['label']
    
    # Initialize emotion as neutral
    emotion = "neutral"
    
    # Map sentiment labels to emotions
    if sentiment_label == 'POSITIVE':
        emotion = 'happy'
    elif sentiment_label == 'NEGATIVE':
        emotion = 'sad'
    
    # Keyword matching for additional emotions
    if "angry" in transcription.lower():
        emotion = "angry"
    return emotion

# Main function to record audio, transcribe it, and detect emotion
def main():
    audio, sr = record_audio()
    ipd.display(ipd.Audio(audio, rate=sr))  # Display the recorded audio
    transcription = transcribe_audio(audio)
    emotion = detect_emotion(transcription)
    return emotion  # Return the detected emotion

if __name__ == "__main__":  
    detected_emotion = main()
    print("Detected emotion:", detected_emotion)
