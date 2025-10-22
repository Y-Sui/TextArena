"""
Quick test to verify API connection works
"""
import os
import textarena as ta

print("Testing API connection...")
print(f"API Key: {os.getenv('OPENROUTER_API_KEY')[:20]}..." if os.getenv('OPENROUTER_API_KEY') else "NOT SET")

try:
    # Try creating an agent
    agent = ta.agents.OpenRouterAgent(
        model_name="anthropic/claude-3.5-haiku",
        verbose=True
    )
    print("✓ Agent created successfully")

    # Try a simple call
    print("\nTesting simple API call...")
    response = agent("Say 'Hello' and nothing else.")
    print(f"✓ API call successful! Response: {response}")

except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your OPENROUTER_API_KEY is valid")
    print("2. Try a different model name")
    print("3. Check your internet connection")
    print("4. Verify your API key has credits")
