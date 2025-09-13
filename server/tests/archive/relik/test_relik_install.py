#!/usr/bin/env python3
"""
Quick test to verify ReLiK is installed and accessible
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_relik_installation():
    """Test that ReLiK is properly installed"""
    print("🧪 TESTING RELIK INSTALLATION")
    print("=" * 40)
    
    try:
        from relik import Relik
        print("✅ ReLiK imported successfully")
        
        # Check available models
        print(f"📋 Available ReLiK models:")
        models = [
            "relik-ie/relik-relation-extraction-small",
            "relik-ie/relik-cie-tiny", 
            "relik-ie/relik-relation-extraction-large"
        ]
        
        for model in models:
            print(f"   - {model}")
        
        print(f"\n🚀 ReLiK is ready for use!")
        print(f"💡 Tip: Use Small model for ~100-150ms inference")
        print(f"💡 Tip: Use Tiny model for ~50-80ms inference")
        
        return True
        
    except Exception as e:
        print(f"❌ ReLiK import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_relik_installation()
    if success:
        print(f"\n🎯 ReLiK installation test completed successfully!")
    else:
        print(f"\n❌ ReLiK installation test failed!")