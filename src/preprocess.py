import torch
import torchaudio
import torchaudio.transforms as T
import os
import soundfile as sf  

def process_audio(input_path, output_folder="processed"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Load using soundfile (safe on Windows)
    data, sample_rate = sf.read(input_path)
    waveform = torch.from_numpy(data).float()
    
    # Ensure [channels, time] format
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T

    # 2. Preprocess
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

    # 3. Save using soundfile (Safe from TorchCodec errors)
    file_name = os.path.basename(input_path)
    output_path = os.path.join(output_folder, f"clean_{file_name}")
    
    # Soundfile expects [time, channels] so we transpose back
    output_data = waveform.T.numpy()
    sf.write(output_path, output_data, target_sample_rate)

    return output_path
