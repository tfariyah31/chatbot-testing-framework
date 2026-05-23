def test_environment_is_working():
    """Verify the test environment is set up correctly."""
    assert True, "Environment is working"

def test_python_version():
    """Verify we are running Python 3.11+"""
    import sys
    major = sys.version_info.major
    minor = sys.version_info.minor
    assert major == 3, f"Expected Python 3, got {major}"
    assert minor >= 11, f"Expected Python 3.11+, got 3.{minor}"
    print(f"\n✅ Python {major}.{minor} confirmed")

def test_required_packages():
    """Verify key packages are installed."""
    import pytest
    import requests
    import dotenv
    print("\n✅ All required packages available")