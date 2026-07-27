"""
Test script for DeepSeek LLM integration
Verify API connection and functionality
"""

from llm_provider import DeepSeek, Ollama, LLMFactory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_deepseek():
    """Test DeepSeek connection and basic functionality"""
    print("=" * 60)
    print("Testing DeepSeek LLM Provider")
    print("=" * 60)
    
    try:
        # Initialize DeepSeek client
        print("\n1. Initializing DeepSeek client...")
        llm = LLMFactory.create_deepseek()
        print("   ✓ DeepSeek client initialized successfully")
        
        # Test basic chat
        print("\n2. Testing basic chat request...")
        message = "Say hello and tell me you're working!"
        response = llm.chat(message)
        print(f"   User: {message}")
        print(f"   DeepSeek: {response}")
        print("   ✓ Basic chat successful")
        
        # Test with system prompt
        print("\n3. Testing with system prompt...")
        system_prompt = "You are an AI assistant specialized in Python programming."
        message = "What is a decorator in Python?"
        response = llm.chat(message, system_prompt=system_prompt)
        print(f"   System: {system_prompt}")
        print(f"   User: {message}")
        print(f"   DeepSeek: {response[:200]}...")
        print("   ✓ System prompt test successful")
        
        # Test streaming
        print("\n4. Testing streaming response...")
        message = "List 3 benefits of using DeepSeek API"
        print(f"   User: {message}")
        print("   DeepSeek (streaming): ", end="", flush=True)
        full_response = ""
        for chunk in llm.stream(message):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n   ✓ Streaming test successful")
        
        print("\n" + "=" * 60)
        print("✓ All DeepSeek tests passed!")
        print("=" * 60)
        
        return True
    
    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("   Make sure DEEPSEEK_API_KEY is set in .env file")
        return False
    
    except Exception as e:
        print(f"\n✗ DeepSeek Error: {e}")
        return False


def test_ollama():
    """Test Ollama connection (if available)"""
    print("\n" + "=" * 60)
    print("Testing Ollama LLM Provider (Optional)")
    print("=" * 60)
    
    try:
        print("\n1. Initializing Ollama client...")
        llm = LLMFactory.create_ollama()
        print("   ✓ Ollama client initialized")
        
        print("\n2. Testing Ollama connection...")
        message = "Say hello!"
        response = llm.chat(message)
        print(f"   User: {message}")
        print(f"   Ollama: {response}")
        print("   ✓ Ollama test successful")
        
        return True
    
    except Exception as e:
        print(f"\n⚠ Ollama not available: {e}")
        print("   (This is OK - Ollama is optional)")
        return False


def test_factory():
    """Test LLMFactory functionality"""
    print("\n" + "=" * 60)
    print("Testing LLM Factory")
    print("=" * 60)
    
    try:
        print("\n1. Creating DeepSeek via factory...")
        deepseek = LLMFactory.create_deepseek()
        print("   ✓ DeepSeek factory working")
        
        print("\n2. Creating Ollama via factory...")
        ollama = LLMFactory.create_ollama()
        print("   ✓ Ollama factory working")
        
        print("\n✓ Factory tests passed!")
        return True
    
    except Exception as e:
        print(f"\n✗ Factory Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LLM PROVIDER TEST SUITE")
    print("=" * 60)
    print(f"DeepSeek API Key: {'✓ Set' if os.getenv('DEEPSEEK_API_KEY') else '✗ Not set'}")
    print(f"Ollama URL: {os.getenv('OLLAMA_API_URL', 'http://localhost:11434')}")
    
    # Run tests
    deepseek_ok = test_deepseek()
    ollama_ok = test_ollama()
    factory_ok = test_factory()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"DeepSeek: {'✓ PASS' if deepseek_ok else '✗ FAIL'}")
    print(f"Ollama: {'✓ PASS' if ollama_ok else '⚠ SKIP'}")
    print(f"Factory: {'✓ PASS' if factory_ok else '✗ FAIL'}")
    print("=" * 60)
    
    exit(0 if deepseek_ok and factory_ok else 1)
