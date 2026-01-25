import os
import json
from preprocess import process_audio
from gemini_client import get_forensic_analysis

def main():
    INPUT_DIR = "data/Fake"
    PROCESSED_DIR = "processed"
    REPORT_FILE = "report.json"
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        return

    # Load existing results if they exist (so we don't lose them)
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r") as f:
            results = json.load(f)
    else:
        results = []

    audio_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.wav', '.mp3'))]

    for filename in audio_files:
        # Skip if already processed
        if any(r['file'] == filename for r in results):
            continue

        raw_path = os.path.join(INPUT_DIR, filename)
        print(f"--- Analyzing: {filename} ---")
        
        try:
            clean_path = process_audio(raw_path, PROCESSED_DIR)
            
            # UPDATED PROMPT: More aggressive toward SOTA AI
            aggressive_prompt = """
            CRITICAL ANALYSIS: This audio may be a high-end AI clone (e.g., ElevenLabs).
            Do not be fooled by 'breathing' or 'natural pauses,' as advanced AI now simulates these.
            
            Look for:
            1. 'Breath Loops': Does every breath sound identical in texture?
            2. 'Spectral Perfection': Is there a total lack of background environmental shifting?
            3. 'Predictable Emotion': Does the excitement feel 'pasted on' rather than building from the previous sentence?
            
            Provide a strict DECISION and CONFIDENCE.
            """
            
            analysis = get_forensic_analysis(clean_path) # Pass prompt if you modified the client
            
            # Save immediately
            results.append({"file": filename, "analysis": analysis})
            with open(REPORT_FILE, "w") as f:
                json.dump(results, f, indent=4)
                
            print(f"Result Saved for {filename}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()