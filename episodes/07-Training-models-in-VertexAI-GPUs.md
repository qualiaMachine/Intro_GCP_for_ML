---
title: "Training Models in Vertex AI: PyTorch Example"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- When should you consider a GPU (or TPU) instance for PyTorch training in Vertex AI, and what are the trade‑offs for small vs. large workloads?
- How do you launch a script‑based training job and write **all** artifacts (model, metrics, logs) next to each other in GCS without deploying a managed model?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Prepare the Titanic dataset and save train/val arrays to compressed `.npz` files in GCS.
- Submit a **CustomTrainingJob** that runs a PyTorch script and explicitly writes outputs to a chosen `gs://…/artifacts/.../` folder.
- Co‑locate artifacts: `model.pt` (or `.joblib`), `metrics.json`, `eval_history.csv`, and `training.log` for reproducibility.
- Choose CPU vs. GPU instances sensibly; understand when distributed training is (not) worth it.

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup (controller notebook)

## Initial setup (controller notebook)

Open a fresh Jupyter notebook in Vertex AI Workbench. Select the **PyTorch** environment (kernel)

Note: local PyTorch is only needed for local tests. Your **Vertex AI job** uses the container specified by `container_uri` (e.g., `pytorch-cpu.2-1` or `pytorch-gpu.2-1`), so it brings its own framework at run time.

```python
from google.cloud import aiplatform, storage
client = storage.Client()
PROJECT_ID = client.project
REGION = "us-central1"
BUCKET_NAME = "sinkorswim-johndoe-titanic" # ADJUST to your bucket's name

print(f"project = {PROJECT_ID}\nregion = {REGION}\nbucket = {BUCKET_NAME}")

# Only used for the SDK's small packaging tarball.
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}/.vertex_staging") # store tar balls in staging folder 
```

## Prepare data as `.npz`

Why `.npz`?

- Smaller, faster I/O than CSV for arrays.
- Natural fit for `torch.utils.data.Dataset` / `DataLoader`.
- One file can hold multiple arrays (`X_train`, `y_train`).
  
```python
import pandas as pd
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Titanic CSV (from local or GCS you've already downloaded to the notebook)
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob("titanic_train.csv")
df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))

# Minimal preprocessing to numeric arrays
sex_enc = LabelEncoder().fit(df["Sex"])            # Fit label encoder on 'Sex' column (male/female)
df["Sex"] = sex_enc.transform(df["Sex"])           # Convert 'Sex' to numeric values (e.g., male=1, female=0)
df["Embarked"] = df["Embarked"].fillna("S")       # Replace missing embarkation ports with most common ('S')
emb_enc = LabelEncoder().fit(df["Embarked"])       # Fit label encoder on 'Embarked' column (S/C/Q)
df["Embarked"] = emb_enc.transform(df["Embarked"]) # Convert embarkation categories to numeric codes
df["Age"] = df["Age"].fillna(df["Age"].median())   # Fill missing ages with median (robust to outliers)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())# Fill missing fares with median to avoid NaNs

X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].values  # Select numeric feature columns as input
y = df["Survived"].values                                                # Target variable (1=survived, 0=did not survive)

scaler = StandardScaler()                                                # Initialize standard scaler for standardization (best practice for neural net training)
X = scaler.fit_transform(X)                                              # Scale features to mean=0, std=1 for stable training

X_train, X_val, y_train, y_val = train_test_split(                       # Split dataset into training and validation sets
    X, y, test_size=0.2, random_state=42)                                # 80% training, 20% validation (fixed random seed)

np.savez("/home/jupyter/train_data.npz", X_train=X_train, y_train=y_train)             # Save training arrays to compressed .npz file
np.savez("/home/jupyter/val_data.npz",   X_val=X_val,   y_val=y_val)                   # Save validation arrays to compressed .npz file

```

We can then upload the files to our GCS bucket.

```python
# Upload to GCS
bucket.blob("data/train_data.npz").upload_from_filename("/home/jupyter/train_data.npz")
bucket.blob("data/val_data.npz").upload_from_filename("/home/jupyter/val_data.npz")
print("Uploaded: gs://%s/data/train_data.npz and val_data.npz" % BUCKET_NAME)
```

## Minimal PyTorch training script (`train_nn.py`)

Find this file in our repo: `Intro_GCP_for_ML/scripts/train_nn.py`. It does three things:
1) loads `.npz` from local or GCS
2) trains a tiny multilayer perceptron (MLP)
3) **writes all outputs side‑by‑side** (model + metrics + eval history + training.log) to the same `--model_out` folder.
   
```python
# GCP_helpers/train_nn.py
import argparse, io, json, os, sys
import numpy as np
import torch, torch.nn as nn
from time import time

#  small helpers for GCS/local I/O 
def _parent_dir(p):
    return p.rsplit("/", 1)[0] if p.startswith("gs://") else (os.path.dirname(p) or ".")

def _write_bytes(path: str, data: bytes):
    if path.startswith("gs://"):
        try:
            import fsspec
            with fsspec.open(path, "wb") as f:
                f.write(data)
        except Exception:
            from google.cloud import storage
            b, k = path[5:].split("/", 1)
            storage.Client().bucket(b).blob(k).upload_from_string(data)
    else:
        os.makedirs(_parent_dir(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

def _write_text(path: str, text: str):
    _write_bytes(path, text.encode("utf-8"))

#  tiny MLP 
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s: x.write(d); x.flush()
    def flush(self):
        for x in self.s: x.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--model_out", required=True, help="gs://…/artifacts/.../model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # All artifacts will sit next to model_out
    model_path = args.model_out
    art_dir = _parent_dir(model_path)

    # capture stdout/stderr
    buf = io.StringIO()
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, buf)
    sys.stderr = _Tee(sys.stderr, buf)
    log_path = f"{art_dir}/training.log"

    try:
        # Load npz (supports gs:// via fsspec)
        def _npz_load(p):
            if p.startswith("gs://"):
                import fsspec
                with fsspec.open(p, "rb") as f:
                    by = f.read()
                return np.load(io.BytesIO(by))
            else:
                return np.load(p)
        train = _npz_load(args.train)
        val   = _npz_load(args.val)
        Xtr, ytr = train["X_train"].astype("float32"), train["y_train"].astype("float32")
        Xva, yva = val["X_val"].astype("float32"),   val["y_val"].astype("float32")

        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).view(-1,1).to(device)
        Xva_t = torch.from_numpy(Xva).to(device)
        yva_t = torch.from_numpy(yva).view(-1,1).to(device)

        model = MLP(Xtr.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = nn.BCELoss()

        hist = []
        t0 = time()
        for ep in range(1, args.epochs+1):
            model.train()
            opt.zero_grad()
            pred = model(Xtr_t)
            loss = loss_fn(pred, ytr_t)
            loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xva_t), yva_t).item()
            hist.append(val_loss)
            if ep % 10 == 0 or ep == 1:
                print(f"epoch={ep} val_loss={val_loss:.4f}")
        print(f"Training time: {time()-t0:.2f}s on {device}")

        # save model
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Saved model: {model_path}")

        # metrics.json and eval_history.csv
        import json
        metrics = {
            "final_val_loss": float(hist[-1]) if hist else None,
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "train_rows": int(Xtr.shape[0]),
            "val_rows": int(Xva.shape[0]),
            "features": list(range(Xtr.shape[1])),
            "model_uri": model_path,
            "device": str(device),
        }
        from io import StringIO
        _write_text(f"{art_dir}/metrics.json", json.dumps(metrics, indent=2))
        csv = "iter,val_loss\n" + "\n".join(f"{i+1},{v}" for i, v in enumerate(hist))
        _write_text(f"{art_dir}/eval_history.csv", csv)
    finally:
        # persist log and restore streams
        try:
            _write_text(log_path, buf.getvalue())
        except Exception as e:
            print(f"[WARN] could not write log: {e}")
        sys.stdout, sys.stderr = orig_out, orig_err

if __name__ == "__main__":
    main()

```

## Launch the training job (no base_output_dir)

```python
RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
ARTIFACT_DIR = f"gs://{BUCKET_NAME}/artifacts/pytorch/{RUN_ID}"
MODEL_URI = f"{ARTIFACT_DIR}/model.pt"   # model + metrics + logs will live here together

job = aiplatform.CustomTrainingJob(
    display_name=f"pytorch_nn_{RUN_ID}",
    script_path="GCP_helpers/train_nn.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-1:latest",  # or pytorch-gpu.2-1
    requirements=["torch", "numpy", "fsspec", "gcsfs"],
)

job.run(
    args=[
        f"--train=gs://{BUCKET_NAME}/data/train_data.npz",
        f"--val=gs://{BUCKET_NAME}/data/val_data.npz",
        f"--epochs=200",
        f"--learning_rate=0.001",
        f"--model_out={MODEL_URI}",   # drives where *all* artifacts go
    ],
    replica_count=1,
    machine_type="n1-standard-4",  # CPU fine for small datasets
    sync=True,
)

print("Artifacts folder:", ARTIFACT_DIR)
```

**What you’ll see in `gs://…/artifacts/pytorch/<RUN_ID>/`:**
- `model.pt` — PyTorch weights (`state_dict`).
- `metrics.json` — final val loss, hyperparameters, dataset sizes, device, model URI.
- `eval_history.csv` — per‑epoch validation loss (for plots/regression checks).
- `training.log` — complete stdout/stderr for reproducibility and debugging.

## Optional: GPU training

For larger models or heavier data:

```python
job = aiplatform.CustomTrainingJob(
    display_name=f"pytorch_nn_gpu_{RUN_ID}",
    script_path="GCP_helpers/train_nn.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
    requirements=["torch", "numpy", "fsspec", "gcsfs"],
)

job.run(
    args=[
        f"--train=gs://{BUCKET_NAME}/data/train_data.npz",
        f"--val=gs://{BUCKET_NAME}/data/val_data.npz",
        f"--epochs=200",
        f"--learning_rate=0.001",
        f"--model_out={MODEL_URI}",
    ],
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    sync=True,
)
```

GPU tips:
- On small problems, GPU startup/transfer overhead can erase speedups—benchmark before you scale.
- Stick to a single replica unless your batch sizes and dataset really warrant data parallelism.

## Distributed training (when to consider)

- **Data parallelism** (DDP) helps when a single GPU is saturated by batch size/throughput. For most workshop‑scale models, a single machine/GPU is simpler and cheaper.
- **Model parallelism** is for very large networks that don’t fit on one device—overkill for this lesson.

## Monitoring jobs & finding outputs

- Console → Vertex AI → Training → Custom Jobs → your run → “Output directory” shows the container logs and the environment’s `AIP_MODEL_DIR`.
- Your script writes **model + metrics + eval history + training.log** next to `--model_out`, e.g., `gs://<bucket>/artifacts/pytorch/<RUN_ID>/`.

::::::::::::::::::::::::::::::::::::: keypoints

- Use **CustomTrainingJob** with a prebuilt PyTorch container; let your script control outputs via `--model_out`.
- Keep artifacts **together** (model, metrics, history, log) in one folder for reproducibility.
- `.npz` speeds up loading and plays nicely with PyTorch.
- Start on CPU for small datasets; use GPU only when profiling shows a clear win.
- Skip `base_output_dir` unless you specifically want Vertex’s default run directory; staging bucket is just for the SDK packaging tarball.

::::::::::::::::::::::::::::::::::::::::::::::::

