#!/usr/bin/env python3
"""
Verify Speaker Profile Storage
Quick diagnostic to see if speaker recognition is working
"""
import sys
import os
from pathlib import Path

print("🔍 Verifying Speaker Recognition System")
print("=" * 70)

# 1. Check profile directory
profile_dir = Path("data/speaker_profiles")
print(f"\n[1] Profile Directory: {profile_dir.absolute()}")

if profile_dir.exists():
    print(f"    ✅ Directory exists")
    
    # List all subdirectories and files
    print(f"\n    Contents:")
    for item in sorted(profile_dir.rglob("*")):
        if item.is_file():
            size_kb = item.stat().st_size / 1024
            print(f"      📄 {item.relative_to(profile_dir)} ({size_kb:.1f} KB)")
        elif item.is_dir() and item != profile_dir:
            print(f"      📁 {item.relative_to(profile_dir)}/")
    
    # Count profile files
    pt_files = list(profile_dir.rglob("*.pt"))
    json_files = list(profile_dir.rglob("*.json"))
    
    print(f"\n    Summary:")
    print(f"      Speaker profiles (.pt): {len(pt_files)}")
    print(f"      Metadata files (.json): {len(json_files)}")
    
    if len(pt_files) > 0:
        print(f"\n    ✅ PROFILES FOUND!")
        for pf in pt_files:
            print(f"       → {pf.name}")
    else:
        print(f"\n    ⚠️  NO PROFILES FOUND")
        print(f"       This means enrollment hasn't completed yet")
else:
    print(f"    ❌ Directory doesn't exist")
    print(f"       The system hasn't been initialized")

# 2. Check test profile directories (from our tests)
print(f"\n[2] Test Profile Directories:")
test_dirs = [
    "data/test_speaker_profiles",
    "data/test_speaker_profiles_3",
    "data/test_emotion_profiles",
    "data/test_prosody_profiles",
]

for test_dir in test_dirs:
    td = Path(test_dir)
    if td.exists():
        pt_files = list(td.rglob("*.pt"))
        if pt_files:
            print(f"    ✅ {test_dir}: {len(pt_files)} profiles")

# 3. Check configuration
print(f"\n[3] Configuration Check:")
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "AUDIO_INTELLIGENCE_ENABLED" in line:
                print(f"    {line.strip()}")
            elif "SPEAKER_PROFILE_DIR" in line:
                print(f"    {line.strip()}")
            elif "AUTO_ENROLL" in line:
                print(f"    {line.strip()}")
            elif "REQUIRE_CONSENT" in line:
                print(f"    {line.strip()}")
            elif "PRIVACY_MODE" in line:
                print(f"    {line.strip()}")

# 4. Check recent logs
print(f"\n[4] Recent Enrollment Activity:")
log_file = Path("data/logs.log")
if log_file.exists():
    with open(log_file) as f:
        lines = f.readlines()
        
    # Look for enrollment events
    enrollment_lines = [
        line for line in lines[-500:]
        if "Enrolled:" in line or "Auto-enrolled:" in line or "Speaker_" in line
    ]
    
    if enrollment_lines:
        print(f"    ✅ Found {len(enrollment_lines)} enrollment events:")
        for line in enrollment_lines[-5:]:  # Last 5
            print(f"       {line.strip()}")
    else:
        print(f"    ⚠️  No enrollment events in recent logs")
        
    # Look for audio processing
    audio_lines = [
        line for line in lines[-100:]
        if "[AudioIntel]" in line and ("Unknown speaker" in line or "Sample" in line)
    ]
    
    if audio_lines:
        print(f"\n    Audio processing detected:")
        for line in audio_lines[-3:]:
            print(f"       {line.strip()}")
    else:
        print(f"\n    ⚠️  No audio processing detected")

# 5. Diagnosis
print(f"\n" + "=" * 70)
print(f"📊 DIAGNOSIS:")

profile_exists = len(list(profile_dir.rglob("*.pt"))) > 0 if profile_dir.exists() else False

if profile_exists:
    print(f"✅ Speaker recognition IS WORKING!")
    print(f"   Your voice has been enrolled and profiles are saved.")
    print(f"   Location: {profile_dir.absolute()}")
elif profile_dir.exists():
    print(f"⚠️  System initialized but NO enrollment yet")
    print(f"   Possible reasons:")
    print(f"   1. Haven't spoken 3 utterances yet")
    print(f"   2. Privacy mode blocking enrollment")
    print(f"   3. Audio quality too inconsistent")
    print(f"   4. Bot not running with audio intelligence enabled")
else:
    print(f"❌ Speaker recognition NOT working")
    print(f"   Possible reasons:")
    print(f"   1. Audio intelligence not enabled in .env")
    print(f"   2. Bot never started with the feature")
    print(f"   3. Profile directory path incorrect")

print(f"\n💡 Next Steps:")
if not profile_exists:
    print(f"   1. Make sure AUDIO_INTELLIGENCE_ENABLED=true in .env")
    print(f"   2. Restart bot: python bot.py")
    print(f"   3. Speak 3 clear sentences")
    print(f"   4. Run this script again to verify")
else:
    print(f"   1. Check profile files with: ls -la {profile_dir}")
    print(f"   2. Profiles are working - test recognition by speaking again")

print(f"\n" + "=" * 70)
