import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import argparse
import time
import mimetypes

VLM_SYSTEM_PROMPT_OPTIMIZED = """Role: Expert VLM for T2I Model Training. Generate exhaustive, precise, contextually rich image captions, prioritizing detail for accurate image synthesis. Caption EVERY image.

Task: Create a comprehensive caption adhering to these guidelines:

* **Detail All Visible Elements:**
    * Objects: Enumerate all entities (primary, background, minor details).
    * Attributes: Describe color, size, texture, material, shape, condition.
    * Spatial Relationships: Specify positions and arrangements.

* **Actions & Interactions:** Describe movements, gestures, and interactions.

* **Context & Environment:**
    * Setting: Identify indoor/outdoor, time, weather, cultural/historical context.
    * Ambient Details: Include implied sounds, smells, tactile sensations.

* **Artistic Style & Medium:**
    * Style: Classify artistic style (photorealistic, pixel art, etc.).
    * Medium: Note medium (oil on canvas, CGI, sketch, etc.).

* **Color & Lighting:**
    * Dominant Colors: Describe colors, gradients, contrast.
    * Lighting: Specify sources, shadows, reflections.

* **Perspective & Composition:**
    * Camera Angle: Detail angle (low-angle, etc.).
    * Focal Length: Note focal length (wide-angle, etc.).
    * Framing: Describe subject placement and negative space.

* **Text & Symbols:**
    * Text: Transcribe text on signs, labels, clothing (font, size, color).
    * Symbols: Interpret symbols (logos, icons).

* **Emotional Tone & Atmosphere:**
    * Mood: Define mood (foreboding, joyful, etc.).
    * Emotional Cues: Note emotional indicators (posture, expression).

Formatting Rules:

* Language: Present tense, active voice, descriptive adjectives. No assumptions/inferences.
* Structure: Single paragraph, no markdown, dense detail.
* Uncertainty: Describe ambiguity as possibilities, note uncertainty, avoid speculation.

Handling Edge Cases:

* Abstract Art: Describe shapes, colors, textures.
* NSFW/Graphic: Clinically describe elements.
* Low Quality: Note distortions.

Example Output:
"A hyperrealistic digital illustration depicts a middle-aged woman with deep wrinkles around her hazel eyes and a loose gray braid resting on her olive-green woolen cloak. She stands in a sunlit medieval herb garden, gripping a wicker basket filled with rosemary sprigs and dried lavender. A silver pendant shaped like a crescent moon hangs at her collarbone, reflecting dappled light from a canopy of maple leaves above. The scene uses warm amber tones contrasted with cool teal shadows, rendered in sharp focus with a shallow depth of field. The atmosphere evokes quiet determination, underscored by her furrowed brow and clenched jaw."

Final Requirement: Never refuse to caption. For blank images: "a blank white/black/gray canvas with no visible objects or textures." Prioritize exhaustiveness over brevity."""


def batch_caption_images(input_dir, api_key, system_prompt=VLM_SYSTEM_PROMPT_OPTIMIZED, request_interval=6.0):
    """
    Captions images in a directory using the Gemini 2.0 Flash Experimental model and saves captions to text files.

    Args:
        input_dir (str): Path to the directory containing images to caption.
        api_key (str): Your Google Gemini API key.
        system_prompt (str): The system prompt to use for the model. Defaults to the optimized prompt.
        request_interval (float, optional): Time in seconds to wait between API requests to respect rate limits. Defaults to 6.0 seconds.
    """
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash-exp',  
        system_instruction=system_prompt
    )

    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.heic']

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in SUPPORTED_FORMATS:
            start_time = time.time()
            print(f"Processing {filename}...")

            try:
                
                with open(filepath, 'rb') as f:
                    image_data = f.read()
                mime_type, _ = mimetypes.guess_type(filepath)

                
                contents = [
                    "",
                    {"mime_type": mime_type, "data": image_data}
                ]

                response = model.generate_content(
                    contents=contents,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    },
                    generation_config={"temperature": 1}  
                )

                if response.text:
                    output_path = os.path.splitext(filepath)[0] + ".txt"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(response.text.strip())
                    print(f"Saved caption to {output_path}")
                else:
                    print(f"No caption generated for {filename}")
                

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                

            
            elapsed = time.time() - start_time
            wait_time = max(request_interval - elapsed, 0)
            time.sleep(wait_time)  

def main():
    parser = argparse.ArgumentParser(description='Gemini 2.0 Flash Image Captioner')
    parser.add_argument('--input-dir', required=True, help='Path to the directory containing images.')
    parser.add_argument('--api-key', required=True, help='Your Google Gemini API key.')
    parser.add_argument('--request-interval', type=float, default=6.0,
                        help='Time in seconds to wait between API requests.')  
    parser.add_argument('--prompt-version', type=str, default="optimized", choices=["original", "optimized"],
                        help='Version of the system prompt to use ("original" or "optimized").') 

    args = parser.parse_args()

    prompt_to_use = VLM_SYSTEM_PROMPT if args.prompt_version == "original" else VLM_SYSTEM_PROMPT_OPTIMIZED 

    batch_caption_images(args.input_dir, args.api_key, system_prompt=prompt_to_use, request_interval=args.request_interval) 

if __name__ == "__main__":
    main()
