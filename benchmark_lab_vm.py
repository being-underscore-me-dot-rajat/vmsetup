"""Benchmark a machine (or VM) against the heaviest steps of the NSSF course labs.

Every test copies the most demanding step of a lab, at the same data size, using synthetic data, so it runs
before the course files are on the machine. It reports how long each step takes, and compares with a reference
laptop (Intel i7 laptop, 16 GB RAM, NVIDIA RTX 4050 6 GB) on which every lab was tested.

Run it with the course environment's Python:
    <env>\\Scripts\\python.exe benchmark_lab_vm.py               all tests
    <env>\\Scripts\\python.exe benchmark_lab_vm.py --skip-llm     no Ollama / LLM tests
    <env>\\Scripts\\python.exe benchmark_lab_vm.py --users 5      simulate 5 people using Ollama at the same time
    <env>\\Scripts\\python.exe benchmark_lab_vm.py --only xgboost,bert

Results are printed as a table and saved to benchmark_results_<computer name>.json next to this script.
"""
import argparse
import json
import os
import platform
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
SEED = 42

# Seconds on the reference laptop (filled in from a run on that machine). None = not measured there.
REFERENCE = {}
REFERENCE_FILE = HERE / "benchmark_reference.json"
if REFERENCE_FILE.exists():
    REFERENCE = json.loads(REFERENCE_FILE.read_text())

results = []


def run_test(name, mirrors, function):
    """Run one test, time it, and record the result (or the error) without stopping the others."""
    print(f"\n--- {name}: {mirrors}")
    start = time.perf_counter()
    estimate = None
    try:
        details = function() or ""
        # A test that runs a smaller version of the lab returns (details, estimated seconds for the full lab step).
        if isinstance(details, tuple):
            details, estimate = details
        seconds = time.perf_counter() - start
        status = "ok"
    except Exception as error:
        seconds = time.perf_counter() - start
        details = f"{type(error).__name__}: {error}"
        status = "FAILED"
        traceback.print_exc(limit=1)
    print(f"    {status} in {seconds:.1f}s  {details}")
    # "compare" is what the summary compares with the reference: the full-lab estimate if there is one.
    compare = round(estimate if estimate is not None else seconds, 1)
    results.append({"test": name, "mirrors": mirrors, "seconds": round(seconds, 1), "compare": compare,
                    "status": status, "details": details})


# ---------------------------------------------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------------------------------------------

def system_info():
    info = {"computer": platform.node(), "os": platform.platform(), "python": sys.version.split()[0],
            "cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["free_disk_gb"] = round(psutil.disk_usage(str(HERE)).free / 1e9, 1)
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        else:
            info["gpu"] = "none"
    except ImportError:
        info["gpu"] = "torch not installed"
    return info


# ---------------------------------------------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------------------------------------------

def make_claims(n=8000, n_features=18):
    """A claims-like table: numbers, a few categories, a 9% irregular target (Day 2-3 size)."""
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    X["benefit_type"] = rng.choice(["A", "B", "C", "D", "E", "F", "G"], n)
    X["channel"] = rng.choice(["BRANCH", "ONLINE", "HR"], n)
    signal = X["f0"] * 1.2 + X["f1"] - X["f2"] * 0.8 + (X["channel"] == "ONLINE") * 0.7
    y = (signal + rng.normal(scale=1.5, size=n) > 2.6).astype(int)
    return X, y


# ---------------------------------------------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------------------------------------------

def test_pandas_large():
    """Day 1 Lab 3 / Day 3 Lab 2: load and aggregate 2.4 million contribution rows, CSV vs Parquet."""
    rng = np.random.default_rng(SEED)
    n = 2_400_000
    df = pd.DataFrame({
        "member_no": rng.integers(100_000_000, 100_060_000, n).astype(str),
        "employer_no": ("ER" + pd.Series(rng.integers(100_000, 101_400, n)).astype(str)).values,
        "period": pd.Series(rng.integers(0, 108, n)).map(lambda m: f"{2017 + m // 12}-{m % 12 + 1:02d}").values,
        "gross_wage_tzs": rng.lognormal(13.4, 0.6, n).round(),
    })
    df["employee_share_tzs"] = (df["gross_wage_tzs"] * 0.1).round()
    df["employer_share_tzs"] = df["employee_share_tzs"]
    folder = HERE / "_bench_tmp"
    folder.mkdir(exist_ok=True)

    t = time.perf_counter()
    df.to_csv(folder / "contrib.csv", index=False)
    csv_write = time.perf_counter() - t
    t = time.perf_counter()
    pd.read_csv(folder / "contrib.csv")
    csv_read = time.perf_counter() - t
    t = time.perf_counter()
    df.to_parquet(folder / "contrib.parquet", index=False)
    back = pd.read_parquet(folder / "contrib.parquet")
    parquet_rw = time.perf_counter() - t

    t = time.perf_counter()
    back["total"] = back["employee_share_tzs"] + back["employer_share_tzs"]
    back.groupby(["employer_no", "period"]).agg(members=("member_no", "nunique"), total=("total", "sum"))
    back.groupby("member_no").agg(months=("period", "count"), wage=("gross_wage_tzs", "mean"))
    groupby = time.perf_counter() - t

    memory_mb = back.memory_usage(deep=True).sum() / 1e6
    return (f"CSV write {csv_write:.1f}s, CSV read {csv_read:.1f}s, Parquet write+read {parquet_rw:.1f}s, "
            f"group-bys {groupby:.1f}s, table in memory {memory_mb:,.0f} MB")


def test_dask():
    """Day 3 Lab 2: the same aggregation with Dask, in parallel over partitions."""
    import dask
    import dask.dataframe as dd
    path = HERE / "_bench_tmp" / "contrib.parquet"
    if not path.exists():
        test_pandas_large()
    ddf = dd.read_parquet(path).repartition(npartitions=9)
    ddf["total"] = ddf["employee_share_tzs"] + ddf["employer_share_tzs"]
    plan1 = ddf.groupby(["employer_no", "period"]).agg({"member_no": "count", "total": "sum"})
    plan2 = ddf.groupby("member_no")["employer_no"].nunique()
    dask.compute(plan1, plan2)
    return f"{ddf.npartitions} partitions"


def test_xgboost_backtest():
    """Day 2 Lab 3: four yearly backtests of XGBoost + logistic regression, plus permutation importance."""
    from sklearn.compose import ColumnTransformer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    X, y = make_claims()
    numeric = [c for c in X.columns if c.startswith("f")]
    categorical = ["benefit_type", "channel"]
    fits = 0
    for cut in [3000, 4000, 5200, 6400]:
        X_tr, y_tr, X_te, y_te = X.iloc[:cut], y.iloc[:cut], X.iloc[cut:cut + 1300], y.iloc[cut:cut + 1300]
        prep = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(), categorical)])
        xgb = Pipeline([("prep", prep), ("model", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                                                subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                                                                random_state=SEED, eval_metric="logloss"))])
        xgb.fit(X_tr, y_tr)
        prep2 = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(), categorical)])
        Pipeline([("prep", prep2), ("model", LogisticRegression(max_iter=2000))]).fit(X_tr, y_tr)
        fits += 2
    permutation_importance(xgb, X_te, y_te, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=1)
    return f"{fits} model fits + permutation importance on {X.shape[1]} columns"


def test_xgboost_large():
    """Enterprise scale: XGBoost on 500,000 rows x 30 features, using every CPU core (headroom check)."""
    from xgboost import XGBClassifier
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(500_000, 30)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(size=500_000) > 1.5).astype(int)
    XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, tree_method="hist").fit(X, y)
    return "500k rows, 300 trees, all cores"


def test_clustering():
    """Day 2 Lab 2: k-means for k = 2..8 with silhouette scores, plus hierarchical clustering on 2,000 rows."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(SEED)
    centres = rng.normal(scale=3, size=(4, 8))
    X = np.vstack([c + rng.normal(size=(3750, 8)) for c in centres])
    sample = rng.choice(len(X), 3000, replace=False)
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        silhouette_score(X[sample], km.labels_[sample])
    KMeans(n_clusters=4, n_init=20, random_state=SEED).fit(X)
    fcluster(linkage(X[sample[:2000]], method="ward"), t=4, criterion="maxclust")
    return "15,000 members x 8 features"


def test_anomaly():
    """Day 5 Lab 1-3: isolation forest (300 trees), ECOD, and 5-fold out-of-fold XGBoost on 34,000 payments."""
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from xgboost import XGBClassifier
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(34_387, 16))
    y = (rng.random(34_387) < 0.012).astype(int)
    groups = rng.integers(0, 4000, 34_387)
    IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X).score_samples(X)
    try:
        from pyod.models.ecod import ECOD
        ECOD().fit(X)
    except ImportError:
        pass
    cross_val_predict(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=4, eval_metric="logloss"),
                      X, y, groups=groups, cv=GroupKFold(n_splits=5), method="predict_proba")
    return "34k payments"


def test_forecasting():
    """Day 4 Lab 3: rolling-origin backtest, 4 x SARIMA + 4 x Prophet, plus 500 simulated years with Prophet."""
    import logging
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("prophet.plot").disabled = True
    from prophet import Prophet
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    months = pd.date_range("2017-03-01", periods=106, freq="MS")
    rng = np.random.default_rng(SEED)
    y = pd.Series(np.exp(1.0 + 0.007 * np.arange(106) + 0.2 * (months.month == 6) + rng.normal(scale=0.04, size=106)),
                  index=months)
    for origin in [58, 70, 82, 94]:
        train = y.iloc[:origin]
        SARIMAX(np.log(train), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).get_forecast(12)
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.8)
        m.fit(pd.DataFrame({"ds": train.index, "y": np.log(train.values)}))
        m.predict(pd.DataFrame({"ds": pd.date_range(train.index[-1], periods=13, freq="MS")[1:]}))
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, uncertainty_samples=500)
    m.fit(pd.DataFrame({"ds": y.index, "y": np.log(y.values)}))
    m.predictive_samples(pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=12, freq="MS")}))
    return "4 SARIMA + 5 Prophet fits"


def torch_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_torch_mlp():
    """Day 3 Lab 1: train the feed-forward network, 60 epochs over 5,300 claims."""
    import torch
    from torch import nn
    device = torch_device()
    torch.manual_seed(SEED)
    X = torch.randn(5273, 39)
    y = (X[:, 0] + X[:, 1] > 1.5).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=128, shuffle=True)
    model = nn.Sequential(nn.Linear(39, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(60):
        for xb, yb in loader:
            loss = loss_fn(model(xb.to(device)).squeeze(1), yb.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return f"on {device}"


def test_autoencoder():
    """Day 5 Lab 1: autoencoder, 30 epochs over 34,000 payments."""
    import torch
    from torch import nn
    device = torch_device()
    X = torch.randn(34_387, 16)
    loader = torch.utils.data.DataLoader(X, batch_size=256, shuffle=True)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3), nn.ReLU(), nn.Linear(3, 8), nn.ReLU(),
                          nn.Linear(8, 16)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        for xb in loader:
            xb = xb.to(device)
            loss = ((model(xb) - xb) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return f"on {device}"


def fake_messages(n, rng):
    """Short English/Swahili-like messages, about the length of the member feedback (40 tokens)."""
    words = ("claim pension paid not yet employer contributions statement branch app member number madai pensheni "
             "michango mwajiri sijalipwa tafadhali naomba msaada since march weeks portal password verification").split()
    return [" ".join(rng.choice(words, rng.integers(15, 35))) for _ in range(n)]


def test_bert_finetune():
    """Day 6 Lab 2: fine-tune multilingual DistilBERT on 4,270 messages for 1 epoch (the lab does 3 on a GPU)."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)
    gpu = torch.cuda.is_available()
    rng = np.random.default_rng(SEED)
    n = 4270 if gpu else 400                         # on a CPU, time 400 messages and scale up
    texts = fake_messages(n, rng)
    labels = rng.integers(0, 9, n).tolist()
    tok = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    enc = tok(texts, truncation=True, max_length=128)
    dataset = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": labels[i]}
               for i in range(n)]
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=9)
    args = TrainingArguments(output_dir=str(HERE / "_bench_tmp" / "bert"), num_train_epochs=1, per_device_train_batch_size=32,
                             save_strategy="no", eval_strategy="no", logging_strategy="no", fp16=gpu, report_to="none",
                             disable_tqdm=True)
    start = time.perf_counter()
    Trainer(model=model, args=args, train_dataset=dataset, data_collator=DataCollatorWithPadding(tok)).train()
    seconds = time.perf_counter() - start
    per_epoch = seconds * 4270 / n
    return (f"{n} messages in {seconds:.0f}s -> about {per_epoch / 60:.1f} min per epoch of the real lab "
            f"({'GPU' if gpu else 'CPU'})", per_epoch)


def test_lora():
    """Day 8 Lab 2: LoRA fine-tuning of Qwen2.5-0.5B (95 conversations x 5 epochs = 60 steps)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    gpu = torch.cuda.is_available()
    bf16 = gpu and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if gpu else torch.float32)
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to("cuda" if gpu else "cpu")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    rng = np.random.default_rng(SEED)
    rows = []
    for text in fake_messages(95, rng):
        ids = tok(text + " " + text + " " + text, add_special_tokens=False)["input_ids"][:110]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids})
    epochs = 5 if gpu else 1                         # on a CPU, time 1 epoch and scale up
    args = TrainingArguments(output_dir=str(HERE / "_bench_tmp" / "lora"), num_train_epochs=epochs, per_device_train_batch_size=8,
                             learning_rate=2e-4, save_strategy="no", eval_strategy="no", logging_strategy="no",
                             bf16=bf16, fp16=gpu and not bf16, report_to="none", disable_tqdm=True)
    start = time.perf_counter()
    Trainer(model=model, args=args, train_dataset=rows,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    seconds = time.perf_counter() - start
    full = seconds * 5 / epochs
    return f"{epochs} epoch(s) in {seconds:.0f}s -> about {full / 60:.1f} min for the lab's 5 epochs", full


def test_hf_generation():
    """Day 7 Lab 1 / Day 8 Lab 3: generate answers with Qwen2.5-0.5B through Transformers (the lab makes ~70)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gpu = torch.cuda.is_available()
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if gpu else torch.float32)
    model = model.to("cuda" if gpu else "cpu")
    new_tokens = 0
    start = time.perf_counter()
    for question in ["How much is the funeral grant?", "When can I get my pension?", "What is a member number?",
                     "How do I change my bank details?", "Who qualifies for maternity benefit?"]:
        text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=120, min_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        new_tokens += out.shape[1] - inputs["input_ids"].shape[1]
    seconds = time.perf_counter() - start
    return f"{new_tokens / seconds:.0f} tokens/s -> Day 8 Lab 3 (66 answers) about {66 * 120 / (new_tokens / seconds) / 60:.1f} min"


def test_embeddings():
    """Day 9 Lab 1: embed 2,000 passages with multilingual E5 and search them with FAISS."""
    import faiss
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-small", device=torch_device())
    rng = np.random.default_rng(SEED)
    passages = ["passage: " + " ".join(fake_messages(1, rng)) * 4 for _ in range(2000)]
    start = time.perf_counter()
    vectors = model.encode(passages, normalize_embeddings=True, batch_size=64).astype(np.float32)
    seconds = time.perf_counter() - start
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index.search(vectors[:200], 5)
    return f"{len(passages) / seconds:.0f} passages/s"


def ollama_generate(model, prompt, tokens):
    import requests
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "num_predict": tokens, "seed": 42}}
    reply = requests.post(OLLAMA_URL + "/api/generate", json=body, timeout=600).json()
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["eval_count"], reply["eval_duration"] / 1e9


def test_ollama(users):
    """Days 7-9: LLM speed through Ollama, one user, then `users` people asking at the same time."""
    import requests
    try:
        installed = [m["name"] for m in requests.get(OLLAMA_URL + "/api/tags", timeout=5).json()["models"]]
    except Exception:
        raise RuntimeError("Ollama isn't running (start the Ollama app or `ollama serve`)")
    parts = []
    prompt = "Explain in detail how a social security fund calculates an old age pension, step by step."
    for model in ["gemma2:2b", "qwen2.5:1.5b"]:
        if model not in installed:
            parts.append(f"{model}: not pulled")
            continue
        ollama_generate(model, "Hello", 5)                      # load the model into memory first
        tokens, seconds = ollama_generate(model, prompt, 250)
        single = tokens / seconds

        # Several people at once: each thread sends one request; measure the total throughput.
        replies = []

        def one_user():
            start = time.perf_counter()
            n, _ = ollama_generate(model, prompt, 250)
            replies.append((n, time.perf_counter() - start))

        threads = [threading.Thread(target=one_user) for _ in range(users)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - start
        total = sum(n for n, _ in replies)
        slowest = max(s for _, s in replies)
        parts.append(f"{model}: {single:.0f} tok/s for one user; {users} users at once: {total / wall:.0f} tok/s total, "
                     f"slowest answer {slowest:.0f}s")
    return " | ".join(parts)


# ---------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-llm", action="store_true", help="skip the Ollama test")
    parser.add_argument("--skip-gpu", action="store_true", help="skip the deep-learning / transformer tests")
    parser.add_argument("--users", type=int, default=5, help="people using Ollama at the same time (default 5)")
    parser.add_argument("--only", default="", help="comma-separated test names to run, e.g. xgboost,bert")
    args = parser.parse_args()

    info = system_info()
    print("Machine:")
    for key, value in info.items():
        print(f"  {key:15s} {value}")

    tests = [
        ("pandas_2.4M_rows", "Day 1 Lab 3 / Day 3 Lab 2 - large table, CSV vs Parquet, group-bys", test_pandas_large),
        ("dask", "Day 3 Lab 2 - parallel aggregation with Dask", test_dask),
        ("xgboost_backtest", "Day 2 Lab 3 - backtest + permutation importance", test_xgboost_backtest),
        ("xgboost_500k", "Headroom - XGBoost on 500k rows, all cores", test_xgboost_large),
        ("clustering", "Day 2 Lab 2 - k-means k=2..8, silhouette, hierarchical", test_clustering),
        ("anomaly", "Day 5 Labs 1-3 - isolation forest, ECOD, out-of-fold XGBoost", test_anomaly),
        ("forecasting", "Day 4 Lab 3 - SARIMA + Prophet backtest, 500 simulated years", test_forecasting),
    ]
    if not args.skip_gpu:
        tests += [
            ("torch_mlp", "Day 3 Lab 1 - feed-forward network, 60 epochs", test_torch_mlp),
            ("autoencoder", "Day 5 Lab 1 - autoencoder, 30 epochs", test_autoencoder),
            ("bert_finetune", "Day 6 Lab 2 - fine-tune DistilBERT (1 epoch)", test_bert_finetune),
            ("lora_finetune", "Day 8 Lab 2 - LoRA fine-tune Qwen2.5-0.5B", test_lora),
            ("hf_generation", "Day 7 Lab 1 / Day 8 Lab 3 - text generation with Transformers", test_hf_generation),
            ("embeddings", "Day 9 Lab 1 - E5 embeddings + FAISS", test_embeddings),
        ]
    if not args.skip_llm:
        tests.append(("ollama", f"Days 7-9 - Ollama LLMs, 1 user and {args.users} users at once",
                      lambda: test_ollama(args.users)))

    if args.only:
        wanted = [w.strip() for w in args.only.split(",")]
        tests = [t for t in tests if any(w in t[0] for w in wanted)]

    total_start = time.perf_counter()
    for name, mirrors, function in tests:
        run_test(name, mirrors, function)
    total = time.perf_counter() - total_start

    # Summary table, with the reference laptop's time and the ratio (above 1 = this machine is slower).
    print("\n" + "=" * 110)
    print(f"{'test':20s} {'seconds':>8s} {'compare':>8s} {'reference':>10s} {'ratio':>6s}  status / details")
    print("-" * 110)
    for r in results:
        ref = REFERENCE.get("compare", {}).get(r["test"])
        ref_text = f"{ref:.1f}" if ref else "-"
        ratio = f"{r['compare'] / ref:.1f}x" if ref and r["status"] == "ok" else "-"
        print(f"{r['test']:20s} {r['seconds']:8.1f} {r['compare']:8.1f} {ref_text:>10s} {ratio:>6s}  "
              f"{r['status']}  {r['details'][:140]}")
    print("-" * 110)
    print(f"total {total / 60:.1f} minutes")
    if REFERENCE:
        print(f"reference machine: {REFERENCE.get('machine', {}).get('cpu', '?')}, "
              f"{REFERENCE.get('machine', {}).get('gpu', '?')}, {REFERENCE.get('machine', {}).get('ram_gb', '?')} GB RAM")
    print("compare: the time used for the ratio (for BERT and LoRA on a CPU: the estimated time of the full lab step).")
    print("ratio: 1.0x = as fast as the reference laptop; 2.0x = twice as slow.")

    out = HERE / f"benchmark_results_{platform.node()}.json"
    out.write_text(json.dumps({"machine": info, "total_seconds": round(total, 1), "results": results,
                               "compare": {r["test"]: r["compare"] for r in results}}, indent=2))
    print("saved", out)

    # Clean up the temporary files.
    import shutil
    shutil.rmtree(HERE / "_bench_tmp", ignore_errors=True)


if __name__ == "__main__":
    main()
