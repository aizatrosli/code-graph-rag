#!/usr/bin/env python3
"""
Comprehensive DeepSeek integration validation script.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_provider_detection():
    """Test DeepSeek provider detection."""
    print("🔍 Testing DeepSeek provider detection...")
    
    from codebase_rag.config import detect_provider_from_model
    
    # Test DeepSeek models
    test_cases = [
        ("deepseek-chat", "deepseek"),
        ("deepseek-coder", "deepseek"),
        ("deepseek-v2", "deepseek"),
        ("azure-gpt-4o-mini", "azure"),
        ("gpt-4", "openai"),
        ("vllm-llama3", "vllm"),
        ("llama3", "local"),
    ]
    
    for model, expected_provider in test_cases:
        actual_provider = detect_provider_from_model(model)
        assert actual_provider == expected_provider, f"Expected {expected_provider} for {model}, got {actual_provider}"
        print(f"  ✓ {model} -> {actual_provider}")
    
    print("✅ Provider detection working correctly")

def test_config_validation():
    """Test DeepSeek configuration validation."""
    print("🔧 Testing DeepSeek configuration validation...")
    
    # Set DeepSeek environment variables
    os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
    os.environ["DEEPSEEK_ENDPOINT"] = "https://api.deepseek.com/v1"
    
    from codebase_rag.config import AppConfig
    
    # Create fresh config instance
    config = AppConfig()
    
    # Test configuration values
    assert config.DEEPSEEK_API_KEY == "test_deepseek_key"
    assert str(config.DEEPSEEK_ENDPOINT) == "https://api.deepseek.com/v1"
    assert config.DEEPSEEK_ORCHESTRATOR_MODEL_ID == "deepseek-chat"
    assert config.DEEPSEEK_CYPHER_MODEL_ID == "deepseek-chat"
    
    print("✅ DeepSeek configuration validation passed")

def test_model_instantiation():
    """Test DeepSeek model instantiation."""
    print("🤖 Testing DeepSeek model instantiation...")
    
    from langchain_openai import ChatOpenAI
    
    # Test creating DeepSeek models with different configurations
    models = [
        ("deepseek-chat", "General chat model"),
        ("deepseek-coder", "Code-focused model"),
        ("deepseek-v2", "Latest model version"),
    ]
    
    for model_name, description in models:
        try:
            llm = ChatOpenAI(
                model=model_name,
                api_key="test_key",
                base_url="https://api.deepseek.com/v1",
                temperature=0.7,
            )
            print(f"  ✓ {model_name}: {description}")
        except Exception as e:
            print(f"  ❌ {model_name}: Failed to instantiate - {e}")
            return False
    
    print("✅ DeepSeek model instantiation successful")
    return True

def test_env_file_example():
    """Test .env file example contains DeepSeek configuration."""
    print("📝 Testing .env.example file...")
    
    env_example_path = Path(".env.example")
    if env_example_path.exists():
        content = env_example_path.read_text()
        
        required_vars = [
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_ENDPOINT",
            "DEEPSEEK_ORCHESTRATOR_MODEL_ID",
            "DEEPSEEK_CYPHER_MODEL_ID",
        ]
        
        for var in required_vars:
            if var in content:
                print(f"  ✓ {var} found in .env.example")
            else:
                print(f"  ❌ {var} missing from .env.example")
                return False
        
        print("✅ .env.example contains DeepSeek configuration")
        return True
    else:
        print("  ❌ .env.example file not found")
        return False

def test_readme_documentation():
    """Test README contains DeepSeek documentation."""
    print("📚 Testing README documentation...")
    
    readme_path = Path("README.md")
    if readme_path.exists():
        try:
            # Try UTF-8 first, then fallback to other encodings
            content = readme_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = readme_path.read_text(encoding='utf-8-sig')
            except UnicodeDecodeError:
                content = readme_path.read_text(encoding='latin-1')
        
        if "DeepSeek" in content and "deepseek" in content.lower():
            print("  ✓ DeepSeek mentioned in README")
            print("✅ README documentation updated")
            return True
        else:
            print("  ❌ DeepSeek not found in README")
            return False
    else:
        print("  ❌ README.md file not found")
        return False

def run_comprehensive_tests():
    """Run all DeepSeek integration tests."""
    print("🚀 Starting comprehensive DeepSeek integration tests...\n")
    
    tests = [
        test_provider_detection,
        test_config_validation,
        test_model_instantiation,
        test_env_file_example,
        test_readme_documentation,
    ]
    
    failed_tests = []
    
    for test_func in tests:
        try:
            result = test_func()
            if result is False:
                failed_tests.append(test_func.__name__)
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}\n")
            failed_tests.append(test_func.__name__)
    
    if not failed_tests:
        print("🎉 All DeepSeek integration tests passed!")
        print("\n🛠️ DeepSeek Setup Summary:")
        print("1. Provider detection: ✅ Working")
        print("2. Configuration: ✅ Working")
        print("3. Model instantiation: ✅ Working")
        print("4. Documentation: ✅ Updated")
        print("\n📝 Next steps:")
        print("• Get your API key from https://platform.deepseek.com/")
        print("• Add DEEPSEEK_API_KEY to your .env file")
        print("• Use models with 'deepseek-' prefix (e.g., deepseek-chat)")
        return True
    else:
        print(f"❌ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
