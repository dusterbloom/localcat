<#
 LocalCat Dev Bootstrap (Windows PowerShell)
 - Ensure Python venv + Node deps
 - Optionally warm models
 - Start server and client
 - Open http://localhost:3000
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Server = Join-Path $Root 'server'
$Client = Join-Path $Root 'client'

function Have-Cmd($name) {
  $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Ensure-Python {
  if (-not (Have-Cmd 'py')) { throw 'Python launcher (py) not found. Install Python 3.12+.' }
  $ver = (& py -V) -replace 'Python\s+', ''
  Write-Host "Python $ver detected"
}

function Ensure-Node {
  if (-not (Have-Cmd 'node') -or -not (Have-Cmd 'npm')) { throw 'Node.js and npm are required.' }
}

function Setup-Venv {
  Push-Location $Server
  if (-not (Test-Path '.venv')) { & py -3.12 -m venv .venv }
  $pip = Join-Path '.venv' 'Scripts\pip.exe'
  & $pip install --upgrade pip
  if (Test-Path 'requirements.txt') { & $pip install -r requirements.txt }
  if (Test-Path 'requirements-ml.txt') { try { & $pip install -r requirements-ml.txt } catch {} }
  Pop-Location
}

function Install-ClientDeps {
  Push-Location $Client
  if (Test-Path 'package-lock.json') { npm ci } else { npm install }
  Pop-Location
}

function Warm-Models {
  Write-Host 'Warming models (best-effort)…'
  Push-Location $Server
  $python = Join-Path '.venv' 'Scripts\python.exe'
  $code = @'
import os

def try_import(msg, fn):
    try:
        fn(); print(f"[warm] OK: {msg}")
    except Exception as e:
        print(f"[warm] skip: {msg}: {e}")

def warm_kokoro():
    from mlx_audio.tts.utils import load_model
    load_model(os.getenv("TTS_MODEL", "mlx-community/Kokoro-82M-bf16"))

try_import("Kokoro TTS model", warm_kokoro)

try:
    import importlib
    if importlib.util.find_spec("parakeet_mlx"):
        from parakeet_mlx import from_pretrained
        from_pretrained(os.getenv("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3"))
        print("[warm] OK: Parakeet STT model")
    else:
        from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
        WhisperSTTServiceMLX(model=MLXModel.MEDIUM)
        print("[warm] OK: Whisper-MLX STT model")
except Exception as e:
    print(f"[warm] skip STT: {e}")
'@
  & $python - << $code
  Pop-Location
}

function Open-Browser {
  Start-Process 'http://localhost:3000'
}

function Start-Server {
  Push-Location $Server
  $py = Join-Path '.venv' 'Scripts\python.exe'
  if (-not (Test-Path '.env') -and (Test-Path '.env.example')) { Copy-Item '.env.example' '.env' }
  Start-Process -FilePath $py -ArgumentList 'bot.py','--host','127.0.0.1','--port','7860' -PassThru | Out-File -FilePath (Join-Path $Root '.server.pid') -Encoding ascii
  Pop-Location
}

function Start-Client {
  Push-Location $Client
  Start-Process -FilePath 'npm' -ArgumentList 'run','dev' -PassThru | Out-File -FilePath (Join-Path $Root '.client.pid') -Encoding ascii
  Pop-Location
}

Ensure-Python
Ensure-Node
Setup-Venv
Install-ClientDeps
if ($env:SKIP_WARM_MODELS -ne '1') { Warm-Models }
Start-Server
Start-Client
Start-Sleep -Seconds 2
Open-Browser
Write-Host '✅ Dev environment running. Open http://localhost:3000'

