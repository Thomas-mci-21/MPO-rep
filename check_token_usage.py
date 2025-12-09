
import os
import json
from openai import OpenAI

# --- Configuration ---
# Load config from the same file the main app uses
CONFIG_FILE = 'configs/model/openai.json'
print(f"Attempting to load configuration from: {CONFIG_FILE}\n")

try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    print(f"✅ Configuration loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Configuration file not found at '{CONFIG_FILE}'")
    exit(1)
except json.JSONDecodeError:
    print(f"❌ Error: Could not decode JSON from '{CONFIG_FILE}'. Please check its format.")
    exit(1)

# --- Get API Details ---
API_BASE = config.get("api_base")
# First, try to get the key from config, if not, fall back to environment variable
API_KEY = config.get("api_key", os.environ.get("OPENAI_API_KEY"))
MODEL_NAME = config.get("model_name")

if not API_BASE:
    print("❌ Error: 'api_base' is missing in the configuration file.")
    exit(1)
if not API_KEY:
    print("❌ Error: 'api_key' is missing in the configuration file and the 'OPENAI_API_KEY' environment variable is not set.")
    exit(1)
if not MODEL_NAME:
    print("❌ Error: 'model_name' is missing in the configuration file.")
    exit(1)

print(f"   - API Base URL: {API_BASE}")
print(f"   - Model Name:   {MODEL_NAME}\n")

# --- API Call ---
try:
    client = OpenAI(
        base_url=API_BASE,
        api_key=API_KEY,
    )

    print("🚀 Sending a test request to the API...")
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a test assistant."}, 
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0,
    )

    print("✅ Request successful. Analyzing response...\n")

    # --- Check for Token Usage ---
    if hasattr(completion, 'usage') and completion.usage is not None:
        print("🎉 SUCCESS: Token usage information was found in the response!")
        print("-------------------------------------------------")
        print(f"Prompt Tokens:     {completion.usage.prompt_tokens}")
        print(f"Completion Tokens: {completion.usage.completion_tokens}")
        print(f"Total Tokens:      {completion.usage.total_tokens}")
        print("-------------------------------------------------")
        print("\nRaw 'usage' object:")
        print(completion.usage)
    else:
        print("⚠️ FAILURE: Token usage information was NOT found in the response.")
        print("The response object did not contain a 'usage' attribute.")
        print("\nRaw response object for debugging:")
        print(completion)

except Exception as e:
    print(f"\n❌ An error occurred during the API call: {e}")
