import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_forensic_analysis(audio_path):
    print(f"Uploading {audio_path}...")
    audio_file = client.files.upload(file=audio_path)

    prompt = """
            CRITICAL ANALYSIS: This audio may be a high-end AI clone (e.g., ElevenLabs).
            Do not be fooled by 'breathing' or 'natural pauses,' as advanced AI now simulates these.
            
            Look for:
            1. 'Breath Loops': Does every breath sound identical in texture?
            2. 'Spectral Perfection': Is there a total lack of background environmental shifting?
            3. 'Predictable Emotion': Does the excitement feel 'pasted on' rather than building from the previous sentence?
            
            Provide a strict DECISION and CONFIDENCE.
            """

    # Generate content using the 'Native Audio' model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, audio_file]
    )

    return response.text

if __name__ == "__main__":
    # Test with a processed file
    # print(get_forensic_analysis("processed/clean_sample.wav"))
    pass



