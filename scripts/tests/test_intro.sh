#!/bin/bash
# Test intro pipeline with forced enrollment
# Usage: ./test_intro.sh

cd "$(dirname "$0")"

echo "🧪 Testing Intro Pipeline..."
echo "   AUDIO_INTEL_FORCE_INTRO=true"
echo "   AUDIO_INTEL_INTRO_PIPELINE=true"
echo "   AUDIO_INTEL_SKIP_FOR_RETURNING=false"
echo ""

# All environment variables on the same line, space-separated
AUDIO_INTEL_FORCE_INTRO=true AUDIO_INTEL_INTRO_PIPELINE=true AUDIO_INTEL_SKIP_FOR_RETURNING=false python bot.py
