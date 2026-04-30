# electricity_ai.py
import ollama
import base64

def extract_units_ai(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    response = ollama.chat(
        model="llama3.2-vision",  # ollama pull llama3.2-vision
        messages=[{
            "role": "user",
            "content": "Look at this electricity bill. Find the total units consumed in kWh. Return ONLY JSON like: {\"units\": 245.5, \"found\": true}. If not found, return {\"found\": false}.",
            "images": [image_data]
        }]
    )
    
    import json, re
    text = response['message']['content']
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"found": False}