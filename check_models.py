
import openai
import toml

try:
    secrets = toml.load(".streamlit/secrets.toml")
    api_key = secrets["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=api_key)
    
    print("Fetching available models...")
    models = client.models.list()
    
    available = [m.id for m in models]
    available.sort()
    
    print(f"Found {len(available)} models.")
    for m in available:
        print(f"- {m}")
            
except Exception as e:
    print(f"Error: {e}")
