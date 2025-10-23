"""
Quick test to verify API connection works
"""
import os
import textarena as ta
from dotenv import load_dotenv

def test_api_connection():
    load_dotenv()  # Load environment variables from .env file if present

    print("Testing API connection...")
    print(f"API Key: {os.getenv('OPENROUTER_API_KEY')[:20]}..." if os.getenv('OPENROUTER_API_KEY') else "NOT SET")
    try:
        # Try creating an agent
        agent = ta.agents.OpenRouterAgent(
            model_name="anthropic/claude-3.5-haiku",
            verbose=True
        )
        print("API connection successful.")
        return True
    except Exception as e:
        print(f"API connection failed: {e}")
        return False
