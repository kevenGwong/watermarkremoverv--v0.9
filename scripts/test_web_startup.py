#!/usr/bin/env python3
"""
Test web application startup
"""

import subprocess
import sys
import time
import signal
from pathlib import Path

def test_existing_debug_app():
    """Test using existing debug app"""
    print("🧪 Testing existing debug app startup...")
    
    # Check if debug app exists
    debug_app_path = Path("watermark_web_app_debug.py")
    if not debug_app_path.exists():
        print(f"❌ Debug app not found: {debug_app_path}")
        return False
    
    print(f"✅ Found debug app: {debug_app_path}")
    
    # Test the --use-existing flag
    try:
        print("🚀 Testing app.py web --use-existing command...")
        
        # Start the process
        process = subprocess.Popen([
            sys.executable, "app.py", "web", 
            "--use-existing", 
            "--port", "8502",
            "--host", "127.0.0.1"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Web app started successfully")
            
            # Terminate the process
            process.terminate()
            process.wait(timeout=5)
            print("✅ Web app terminated cleanly")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Web app failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if 'process' in locals():
            try:
                process.terminate()
            except:
                pass
        return False

def test_direct_streamlit():
    """Test direct streamlit launch"""
    print("\n🧪 Testing direct streamlit launch...")
    
    debug_app_path = Path("watermark_web_app_debug.py")
    if not debug_app_path.exists():
        print(f"❌ Debug app not found: {debug_app_path}")
        return False
    
    try:
        print("🚀 Testing direct streamlit run...")
        
        # Test streamlit command availability
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "--version"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Streamlit is available")
            print(f"Streamlit version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Streamlit not available")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 Testing Web Application Startup")
    print("=" * 50)
    
    tests = [
        ("Direct Streamlit", test_direct_streamlit),
        ("Existing Debug App", test_existing_debug_app),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} CRASHED: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All web startup tests passed!")
        print("\n💡 Usage:")
        print("  python app.py web --use-existing        # Use your existing debug app")
        print("  bash run_debug_app.sh                   # Use your original launcher")
        return 0
    else:
        print(f"\n💥 {failed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())