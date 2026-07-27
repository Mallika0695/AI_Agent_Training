"""
Ollama Integration Example
Local LLM - Free, runs entirely on your machine
"""

from llm_provider import LLMFactory

def main():
    print("=" * 60)
    print("Ollama LLM Integration")
    print("=" * 60)
    
    # Create Ollama client
    print("\nInitializing Ollama client...")
    llm = LLMFactory.create_ollama()
    
    # Test message
    print("Sending test message to Ollama...\n")
    response = llm.chat("What is machine learning? Give a brief answer.")
    
    print("Response:")
    print(response)
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
