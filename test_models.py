import os, requests
from dotenv import load_dotenv

load_dotenv('c:\\folsom-aqi\\.env')
key = os.environ.get("GEMINI_API_KEY")

if not key:
    print("NO KEY")
    exit(1)

models_json = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}").json()
models = [m['name'] for m in models_json.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]

print("TESTING MODELS:", models)

for model in models:
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": "hi"}]}]}
    try:
        req = requests.post(url, json=payload, timeout=5)
        if req.status_code == 200:
            print(f"SUCCESS: {model}")
            exit(0)
        else:
            print(f"FAILED {model}: {req.status_code} {req.text[:80]}")
    except Exception as e:
        print(f"ERROR {model}: {e}")
