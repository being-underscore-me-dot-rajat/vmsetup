<#
Sets up a Windows machine (or VM) for the NSSF Enterprise AI & LLM Engineering course.

What it does, in order:
  1. Installs the software with winget: Python 3.13, Visual Studio Code, Git, Ollama,
     the Microsoft C++ runtime (needed by PyTorch), and optionally Docker Desktop.
  2. Turns on Windows long-path support (needs Administrator; skipped otherwise).
  3. Installs the VS Code extensions: Python, Pylance, Jupyter.
  4. Creates the course Python environment at <CourseRoot>\DAY1\Labs\work\fund-ai\.venv
     (the same place every day's setup cell looks for it) and installs every library from Days 1-10.
     If an NVIDIA GPU is present, PyTorch is switched to its GPU (CUDA) build.
  5. Downloads the spaCy English model, registers the Jupyter kernel "Python (fund-ai)",
     pulls the Ollama models and pre-downloads the Hugging Face models the labs use,
     so nothing large has to download during class.
  6. Checks that everything imports.

Run it from PowerShell, in the VM_Setup folder of the course:
    Set-ExecutionPolicy -Scope Process Bypass
    .\setup_lab_vm.ps1

Options:
    -CourseRoot C:\NSSF        the course folder (default: the folder above this script)
    -EnvPath    C:\fund-ai-venv put the Python environment somewhere else (e.g. a short path)
    -WithDocker                 also install Docker Desktop (Day 10; needs virtualisation on the VM)
    -SkipSoftware              skip step 1 (software already installed)
    -SkipModels                skip the Ollama and Hugging Face downloads (several GB)
Safe to run more than once: anything already installed is skipped.
#>
param(
    [string]$CourseRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$EnvPath = "",
    [switch]$WithDocker,
    [switch]$SkipSoftware,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$OllamaModels = @("qwen2.5:1.5b", "llama3.2:1b", "gemma2:2b", "qwen2.5:0.5b")
$HfModels = @("Qwen/Qwen2.5-0.5B-Instruct", "distilbert-base-multilingual-cased",
              "intfloat/multilingual-e5-small", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

function Step($text) { Write-Host "`n==== $text" -ForegroundColor Cyan }
function Info($text) { Write-Host "  $text" }
function Warn($text) { Write-Host "  WARNING: $text" -ForegroundColor Yellow }

function Refresh-Path {
    # Pick up PATH changes made by the installers, without opening a new window.
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Install-WithWinget($id, $name) {
    $installed = winget list --id $id --exact --accept-source-agreements 2>$null | Select-String -SimpleMatch $id
    if ($installed) {
        Info "$name already installed"
        return
    }
    Info "installing $name ..."
    winget install --id $id --exact --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { Warn "winget could not install $name (exit code $LASTEXITCODE). Install it by hand." }
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

# ---------------------------------------------------------------------------------------------
Step "1. Software"
if ($SkipSoftware) {
    Info "skipped (-SkipSoftware)"
} else {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget isn't available. Install 'App Installer' from the Microsoft Store, then run this script again."
    }
    Install-WithWinget "Python.Python.3.13" "Python 3.13"
    Install-WithWinget "Microsoft.VisualStudioCode" "Visual Studio Code"
    Install-WithWinget "Git.Git" "Git"
    Install-WithWinget "Ollama.Ollama" "Ollama"
    Install-WithWinget "Microsoft.VCRedist.2015+.x64" "Microsoft C++ runtime"
    if ($WithDocker) { Install-WithWinget "Docker.DockerDesktop" "Docker Desktop" }
    Refresh-Path
}

# ---------------------------------------------------------------------------------------------
Step "2. Windows long paths"
$longPaths = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
if ($longPaths -eq 1) {
    Info "already enabled"
} elseif ($isAdmin) {
    Set-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1
    Info "enabled"
} else {
    Warn "long paths are off and this window isn't running as Administrator. Either rerun as Administrator,"
    Warn "or keep the course in a short folder (e.g. C:\NSSF) or use -EnvPath C:\fund-ai-venv."
}

# ---------------------------------------------------------------------------------------------
Step "3. VS Code extensions"
$code = Get-Command code -ErrorAction SilentlyContinue
if (-not $code) {
    $guess = Join-Path $env:LOCALAPPDATA "Programs\Microsoft VS Code\bin\code.cmd"
    if (Test-Path $guess) { $code = Get-Item $guess }
}
if ($code) {
    foreach ($ext in @("ms-python.python", "ms-python.vscode-pylance", "ms-toolsai.jupyter")) {
        Info "extension $ext"
        & $code.Source --install-extension $ext --force | Out-Null
    }
} else {
    Warn "VS Code command 'code' not found; install the Python and Jupyter extensions from inside VS Code."
}

# ---------------------------------------------------------------------------------------------
Step "4. Course Python environment"
if (-not (Test-Path (Join-Path $CourseRoot "DAY1"))) {
    Warn "$CourseRoot doesn't contain DAY1. The environment will still be created; copy the course there later."
}
if ($EnvPath -eq "") { $EnvPath = Join-Path $CourseRoot "DAY1\Labs\work\fund-ai\.venv" }
$py = Join-Path $EnvPath "Scripts\python.exe"
$requirements = Join-Path $PSScriptRoot "requirements-all.txt"

if (Test-Path $py) {
    Info "environment found: $EnvPath"
} else {
    Info "creating $EnvPath"
    New-Item -ItemType Directory -Force (Split-Path $EnvPath) | Out-Null
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { & py -3.13 -m venv $EnvPath } else { & python -m venv $EnvPath }
    if (-not (Test-Path $py)) { throw "Could not create the environment. Is Python 3.13 installed?" }
}

Info "upgrading pip"
& $py -m pip install --quiet --upgrade pip
Info "installing every course library (first time: 10-20 minutes) ..."
& $py -m pip install --quiet -r $requirements
if ($LASTEXITCODE -ne 0) { throw "pip install failed. If the error mentions a long path, see step 2." }

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $cuda = & $py -c "import torch; print(torch.cuda.is_available())"
    if ($cuda -ne "True") {
        Info "NVIDIA GPU found: installing the GPU build of PyTorch (about 2.5 GB) ..."
        & $py -m pip uninstall -y -q torch
        & $py -m pip install --quiet "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
    }
    Info ("PyTorch GPU: " + (& $py -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"))
} else {
    Info "no NVIDIA GPU found: PyTorch will run on the CPU"
}

# ---------------------------------------------------------------------------------------------
Step "5. Models, kernel and downloads"
$spacyOk = & $py -c "import importlib.util; print(importlib.util.find_spec('en_core_web_sm') is not None)"
if ($spacyOk -ne "True") {
    Info "spaCy English model"
    & $py -m spacy download en_core_web_sm --quiet
} else { Info "spaCy English model already installed" }

Info "Jupyter kernel 'Python (fund-ai)'"
& $py -m ipykernel install --user --name fund-ai --display-name "Python (fund-ai)" | Out-Null

if ($SkipModels) {
    Info "model downloads skipped (-SkipModels)"
} else {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollama) {
        try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/version -TimeoutSec 3 | Out-Null }
        catch {
            Info "starting Ollama"
            Start-Process -WindowStyle Hidden ollama serve
            Start-Sleep -Seconds 5
        }
        foreach ($m in $OllamaModels) {
            Info "ollama pull $m"
            & ollama pull $m
        }
    } else {
        Warn "Ollama not found; install it and pull: $($OllamaModels -join ', ')"
    }

    Info "Hugging Face models (about 3 GB the first time) ..."
    $list = ($HfModels | ForEach-Object { "'$_'" }) -join ","
    & $py -c "from huggingface_hub import snapshot_download`nfor m in [$list]:`n    print('   ', m); snapshot_download(m)"
}

# ---------------------------------------------------------------------------------------------
Step "6. Check"
& $py -c @"
import importlib
mods = ['numpy','pandas','sklearn','xgboost','lightgbm','statsmodels','prophet','dask','pyarrow','torch','transformers',
        'peft','sentence_transformers','faiss','spacy','gensim','imblearn','pyod','fastapi','mlflow','langchain_ollama','dvc']
bad = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f'{m}: {e}')
import torch
print('  Python OK;', len(mods) - len(bad), 'of', len(mods), 'libraries import')
print('  PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
for b in bad: print('  FAILED', b)
"@

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Environment: $py"
Write-Host "In VS Code, pick the kernel 'Python (fund-ai)', or Python: Select Interpreter -> the path above."
Write-Host "To measure this machine: $py $(Join-Path $PSScriptRoot 'benchmark_lab_vm.py')"
