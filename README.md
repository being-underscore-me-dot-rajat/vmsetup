# Setting up and sizing a machine for the course

This folder has two tools:

| File | What it does |
|---|---|
| `setup_lab_vm.ps1` (Windows) / `setup_lab_vm.sh` (Ubuntu/Debian) | Installs everything the course needs and pre-downloads the models |
| `requirements-all.txt` | Every Python library used in Days 1-10, with the course's pinned versions |
| `benchmark_lab_vm.py` | Times the heaviest step of each lab, so you can judge whether a machine is fast enough |
| `benchmark_reference.json` | The timings from the laptop every lab was tested on, used for comparison |

## 1. Set up a machine

Copy the whole course folder to the machine first, ideally to a short path such as `C:\NSSF`, then in PowerShell:

```powershell
cd C:\NSSF\VM_Setup
Set-ExecutionPolicy -Scope Process Bypass
.\setup_lab_vm.ps1                 # add -WithDocker for Day 10's containers
```

Run it **as Administrator** if you can: it then also switches on Windows long-path support. It installs:

- **Software (winget):** Python 3.13, Visual Studio Code, Git, Ollama, the Microsoft C++ runtime, optionally Docker Desktop
- **VS Code extensions:** Python, Pylance, Jupyter
- **The course environment** at `DAY1\Labs\work\fund-ai\.venv` (where every day's setup cell looks for it), with all
  libraries; the GPU build of PyTorch if the machine has an NVIDIA GPU
- **The spaCy English model**, and the Jupyter kernel **Python (fund-ai)**
- **Model downloads (about 8 GB):** Ollama models `qwen2.5:1.5b`, `llama3.2:1b`, `gemma2:2b`, `qwen2.5:0.5b`, and the
  Hugging Face models Qwen2.5-0.5B-Instruct, multilingual DistilBERT, multilingual-E5-small and MiniLM

First run: about 30-45 minutes, mostly downloads. It's safe to run again; anything already there is skipped.
Useful options: `-EnvPath C:\fund-ai-venv` (a short path), `-SkipSoftware`, `-SkipModels`.

On Linux: `bash setup_lab_vm.sh` (uses `sudo` for system packages).

Allow about **25 GB of free disk** for the software, environment (about 6 GB with GPU PyTorch) and models.

## 2. Measure a machine

```powershell
C:\NSSF\DAY1\Labs\work\fund-ai\.venv\Scripts\python.exe C:\NSSF\VM_Setup\benchmark_lab_vm.py
```

It takes 4-30 minutes depending on the machine. Options: `--users 5` (how many people use Ollama at the same time),
`--skip-llm`, `--skip-gpu`, `--only bert,lora`.

| Test | Mirrors | Stresses |
|---|---|---|
| `pandas_2.4M_rows` | Day 1 Lab 3, Day 3 Lab 2 | RAM, disk, single-core CPU |
| `dask` | Day 3 Lab 2 | Multi-core CPU |
| `xgboost_backtest`, `xgboost_500k` | Day 2 Lab 3; headroom | Multi-core CPU |
| `clustering` | Day 2 Lab 2 | CPU |
| `anomaly` | Day 5 Labs 1-3 | CPU |
| `forecasting` | Day 4 Lab 3 | CPU (Prophet/Stan) |
| `torch_mlp`, `autoencoder` | Day 3 Lab 1, Day 5 Lab 1 | GPU or CPU |
| `bert_finetune` | Day 6 Lab 2 | **GPU** |
| `lora_finetune` | Day 8 Lab 2 | **GPU** (needs about 3 GB of GPU memory) |
| `hf_generation` | Day 7 Lab 1, Day 8 Lab 3 | GPU |
| `embeddings` | Day 9 Lab 1 | GPU or CPU |
| `ollama` | Days 7-9 | GPU (or CPU), and how it copes with several people at once |

The summary table compares every test with the reference laptop (Intel i7, 8 cores, 16 GB RAM, NVIDIA RTX 4050 6 GB).
`ratio` 1.0x means as fast; 2.0x means twice as slow. For BERT and LoRA without a GPU, the benchmark runs a smaller
version and estimates the full lab step.

### Reference timings (the laptop the labs were tested on)

| Test | Seconds |
|---|---|
| pandas_2.4M_rows | 17.8 |
| dask | 4.1 |
| xgboost_backtest | 3.4 |
| xgboost_500k | 7.6 |
| clustering | 3.2 |
| anomaly | 5.7 |
| forecasting | 8.5 |
| torch_mlp | 13.5 |
| autoencoder | 13.8 |
| bert_finetune (1 epoch) | 19 |
| lora_finetune (5 epochs) | 21 |
| hf_generation | 26.2 (25 tokens/s) |
| embeddings | 11.3 (358 passages/s) |
| ollama | gemma2:2b 85 tokens/s, qwen2.5:1.5b 116 tokens/s for one user; with 5 users at once about 70-95 tokens/s in total, slowest answer 13-18 s |

On the same laptop **without its GPU**, fine-tuning DistilBERT takes about 7 minutes per epoch (22x slower) and the
LoRA fine-tune about 11 minutes. The labs still run (they have a `QUICK_RUN` switch), but Days 6, 8 and 9 become slow.

## 3. Reading the results

- **CPU tests (Days 1-5) at 2x or better**: comfortable. At 3x or worse, those labs feel slow but still finish.
- **No GPU**: everything works, but plan for Days 6-9 to take noticeably longer, and use `QUICK_RUN` in Day 6 Lab 2 and
  Day 8 Lab 2.
- **One VM shared by several participants**: look at the `ollama` line with `--users` set to the number of people.
  Ollama answers requests one after another by default, so the slowest answer grows with the number of people. If it
  goes above about 30 seconds, either give each participant their own machine, use a bigger GPU, or start Ollama with
  `OLLAMA_NUM_PARALLEL=4` (needs more GPU memory). Also check that RAM is at least 8 GB per participant plus 8 GB.
- **Minimum per participant** (tested): 4 cores, 16 GB RAM, 25 GB disk. **Recommended**: 8 cores, 16-32 GB RAM and an
  NVIDIA GPU with 6 GB or more.
