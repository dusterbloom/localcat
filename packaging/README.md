# 🦉 Cucumeu Packaging

This directory contains scripts to package Cucumeu for distribution to beta testers.

## Quick Package Creation

```bash
cd packaging
chmod +x create_cucumeu_package.sh
./create_cucumeu_package.sh
```

This creates `cucumeu-macos-0.1.0-beta.zip` ready to send to testers.

## What's Included

- Complete server and client code
- Automated setup script (downloads models, installs dependencies)
- Simple start script (one command to run)
- Branded README with clear instructions
- Testing guide for beta feedback

## For Testers

Send them the ZIP file with this message:

> Hey! I built an AI assistant called Cucumeu that actually remembers your conversations - unlike ChatGPT which forgets everything. It runs 100% on your Mac (totally private). Takes about 10 minutes to set up. Want to try it and let me know what you think? 🦉

## Package Contents

```
cucumeu-macos-0.1.0-beta/
├── README.md              # User-friendly introduction
├── setup-cucumeu.sh       # One-time setup (8-10 min)
├── start-cucumeu.sh       # Run Cucumeu
├── TEST_INSTRUCTIONS.md   # Beta testing guide
├── server/                # Python backend
└── client/                # Web interface
```

## System Requirements

Testers need:
- Apple Silicon Mac (M1/M2/M3)
- macOS 12+
- Python 3.12+
- Node.js 18+
- LM Studio (free, for running local LLM)

## Support

If testers have issues:
1. Check `server/cucumeu-server.log` for server errors
2. Check `cucumeu-client.log` for client issues
3. Make sure LM Studio is running with a model loaded
4. Ensure port 3000 and 8000 are free

## Next Steps

After beta testing:
1. Collect feedback via TEST_INSTRUCTIONS.md form
2. Fix critical issues
3. Consider Homebrew formula for easier installation
4. Add voice recognition feature
5. Launch on GitHub
