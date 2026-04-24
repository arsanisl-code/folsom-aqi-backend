import os
import requests
from dotenv import load_dotenv

def main():
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")

    if not key:
        print("NO KEY")
        return

    try:
        models_json = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        ).json()
        models = [
            m["name"]
            for m in models_json.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        print("TESTING MODELS:", models)

        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={key}"
            payload = {"contents": [{"parts": [{"text": "hi"}]}]}
            try:
                req = requests.post(url, json=payload, timeout=5)
                if req.status_code == 200:
                    print(f"SUCCESS: {model}")
                    return
                else:
                    print(f"FAILED {model}: {req.status_code} {req.text[:80]}")
            except Exception as e:
                print(f"ERROR {model}: {e}")
    except Exception as e:
        print(f"ERROR fetching models: {e}")


if __name__ == "__main__":
    main()
