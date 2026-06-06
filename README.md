# email-subject-generator

![CI](https://github.com/xavier-oc-programming/email-subject-generator/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Model](https://img.shields.io/badge/model-T5--small-orange)
![Dataset](https://img.shields.io/badge/dataset-AESLC-green)
![Azure](https://img.shields.io/badge/deployed-Azure%20App%20Service-0078D4?logo=microsoftazure)

Generates professional email subject line suggestions from an email body
using T5-small fine-tuned on the AESLC corpus (14,000 Enron email/subject pairs).
Returns five suggestions spanning conservative to creative tone.

**Live demo → [email-subject-gen-xoc.azurewebsites.net](https://email-subject-gen-xoc.azurewebsites.net)**
· **API docs → [/docs](https://email-subject-gen-xoc.azurewebsites.net/docs)**
· **Notebook → notebook.ipynb**

---

## If you're cloning this repo

**Analysis only (no server)**
```bash
git clone https://github.com/xavier-oc-programming/email-subject-generator
cd email-subject-generator
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

**Run the web app locally**
```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:8000
# Note: requires trained model files in models/
```

**Deploy your own instance to Azure**
Fork the repo, create Azure resources (see Deployment section), then add your own `AZURE_CREDENTIALS` secret — it is scoped to the original resource group and is not inherited by forks.

---

## Architecture

The model follows the encoder-decoder (seq2seq) architecture introduced by Sutskever et al. (2014).

**Encoder**
- Embedding layer maps each input character to a 64-dimensional vector
- Single LSTM layer with 256 units reads the email body character by character
- Final hidden state `[h, c]` = the context vector — a fixed-size summary of the entire input

**Decoder (training)**
- Initialized with the encoder's `[h, c]` states
- Uses teacher forcing: at each step, the true previous character is fed as input
- Dense softmax layer predicts the next character probability distribution

**Decoder (inference)**
- Re-uses the trained LSTM and Dense weights
- Runs autoregressively — feeds its own previous output as the next input
- Temperature sampling controls randomness at generation time

**Why separate training and inference models?**
Teacher forcing (training) requires knowing the correct previous character at every step. At inference, no ground truth exists, so the model feeds its own predictions back. Keras requires two separate model graphs to support this.

---

## Dataset

**AESLC — Annotated Enron Subject Line Corpus** (loaded via HuggingFace `datasets`)

- ~14,000 professional email body / subject line pairs
- Splits: train / validation / test

**Preprocessing**
- Body truncated to first 200 characters
- Lowercased, printable ASCII only
- Subject lines: 3–60 characters
- Email bodies: ≥ 20 characters
- Subject lines wrapped with `<` (start) and `>` (end) tokens

---

## Training

```bash
python train.py
```

Runs end to end: loads data, builds vocabulary, encodes sequences, trains with early stopping, saves models and plots to `models/` and `plots/`.

**Callbacks**
- `EarlyStopping(patience=5, monitor=val_loss)` — stops when validation loss stops improving
- `ModelCheckpoint` — saves the best epoch only

**Hyperparameters** (all in `config.py`)

| Parameter | Value |
|-----------|-------|
| Latent dim | 256 |
| Embedding dim | 64 |
| Max body len | 200 chars |
| Max subject len | 60 chars |
| Batch size | 64 |
| Max epochs | 50 |
| Learning rate | 0.001 |

---

## Inference

Temperature controls how conservative or creative the generation is:

| Temperature | Style | Effect |
|-------------|-------|--------|
| < 0.6 | Conservative | Sharpens distribution — picks common patterns |
| 0.6–0.8 | Balanced | Good default for professional subject lines |
| > 0.8 | Creative | Flattens distribution — more varied, less predictable |

`generate_multiple()` returns 5 suggestions spanning the temperature range so the user can choose the tone that fits.

---

## Results

_Fill after training: val_loss, sample generations, training curves._

Sample generations (representative):

| Email body (first 80 chars) | Generated subject |
|-----------------------------|-------------------|
| _run train.py to populate_ | — |

---

## API Reference

**POST /generate**
```json
{
  "body_text": "Following up on our Q3 budget discussion...",
  "n": 5,
  "temperature": null
}
```
Response:
```json
{
  "suggestions": [
    {"subject": "Q3 Budget Follow-Up", "temperature": 0.5, "style": "conservative"},
    ...
  ],
  "body_preview": "Following up on our Q3...",
  "model_info": "Encoder-Decoder LSTM | vocab=98 | latent_dim=256 | val_loss=1.23",
  "n_generated": 5
}
```

**GET /health** — model load status, vocab size, latent dim

**GET /api/model-info** — full model config JSON

**GET /api/examples** — 5 example email bodies with descriptions

---

## File structure

```
email-subject-generator/
├── config.py                  # Single source of truth — all constants
├── preprocess.py              # Data loading, vocabulary, encoding
├── model.py                   # Encoder-decoder LSTM (Keras functional API)
├── train.py                   # Training script — run once locally
├── generator.py               # Inference: generate_subject, generate_multiple
├── main.py                    # FastAPI app
├── conftest.py                # pytest path setup
├── Dockerfile
├── startup.txt                # Azure startup command
├── notebook.ipynb             # Architecture walkthrough
├── README.md
├── requirements.txt
├── portfolio.yaml
├── .gitignore
├── .github/workflows/ci.yml   # CI: test + deploy
├── templates/index.html       # Demo frontend (inline CSS/JS)
├── tests/test_api.py
├── models/                    # Committed after training
│   ├── inference_encoder.keras
│   ├── inference_decoder.keras
│   ├── vocab.json
│   ├── training_history.json
│   └── model_config.json
└── plots/                     # Committed after training
    ├── 01_training_curves.png
    ├── 02_subject_length_distribution.png
    ├── 03_body_length_distribution.png
    └── 04_sample_generations.png
```

---

## Deployment

**One-time Azure setup (F1 free tier)**
```bash
az group create --name email-subject-gen-rg --location westeurope
az appservice plan create --name email-subject-gen-plan \
  --resource-group email-subject-gen-rg --sku F1 --is-linux
az webapp create --name email-subject-gen-xoc \
  --resource-group email-subject-gen-rg \
  --plan email-subject-gen-plan --runtime "PYTHON:3.11"
az webapp config set --name email-subject-gen-xoc \
  --resource-group email-subject-gen-rg \
  --startup-file "gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600"
az webapp config appsettings set --name email-subject-gen-xoc \
  --resource-group email-subject-gen-rg \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

**Manual deploy (zip only app files — not notebooks, data, plots)**
```bash
zip -rq deploy.zip main.py generator.py config.py preprocess.py \
  requirements.txt startup.txt models/ templates/
TOKEN=$(az account get-access-token --query accessToken -o tsv)
curl -X POST \
  "https://email-subject-gen-xoc.scm.azurewebsites.net/api/zipdeploy" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/zip" \
  --data-binary @deploy.zip
```

Use `curl` to the Kudu endpoint — not `az webapp deploy` (hangs on F1 free tier due to polling).

---

## CI/CD

Two GitHub Actions jobs on push to `main`:

1. **test** — installs dependencies, runs `pytest tests/ -v`
2. **deploy** — zips app files, authenticates via `AZURE_CREDENTIALS`, uploads via curl to Kudu

The `AZURE_CREDENTIALS` secret is a service principal scoped to `email-subject-gen-rg`. It is private to this repo and not inherited by forks — fork owners must create their own Azure resources and secret.

**One-time secret setup**
```bash
az ad sp create-for-rbac \
  --name "email-subject-gen-gh-actions" \
  --role contributor \
  --scopes /subscriptions/<sub-id>/resourceGroups/email-subject-gen-rg \
  --sdk-auth | gh secret set AZURE_CREDENTIALS \
  --repo xavier-oc-programming/email-subject-generator
```

---

## Design decisions

**Why encoder-decoder, not a simple classifier?**
Generating a sequence (subject line) conditioned on another sequence (email body) is the classic seq2seq problem. A classifier can only pick from fixed categories; this model generates novel text.

**Why character-level?**
Avoids vocabulary limitations — the model can generate any word in any casing, and never hits an out-of-vocabulary token. Trade-off: slower inference and longer sequences. For 60-character subject lines this is acceptable.

**Why teacher forcing?**
Without teacher forcing, early errors in the decoder cascade into increasingly wrong predictions, making training unstable. Teacher forcing decouples training steps, making them stable and fast. The trade-off (exposure bias) is acceptable for short outputs like subject lines.

**Why temperature range 0.5–0.9?**
Professional subject lines need to be coherent (rules out > 1.0) but not formulaic (rules out < 0.4). The range 0.5–0.9 spans conservative to creative while staying professional.

**Why AESLC?**
Professionally written, short subject lines with matching email bodies — exactly the domain. 14k pairs is sufficient for a character-level model on short outputs.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥ 2.15 | Encoder-decoder LSTM |
| numpy | ≥ 1.24, < 2.0 | Array operations |
| pandas | ≥ 2.0 | Data handling |
| matplotlib | ≥ 3.7 | Training plots |
| datasets | ≥ 2.0 | AESLC loading |
| fastapi | ≥ 0.110 | REST API |
| uvicorn | ≥ 0.27 | ASGI server |
| gunicorn | ≥ 21.0 | Production serving |
| pydantic | ≥ 2.0 | Request/response models |
| jupyter | ≥ 1.0 | Notebook |
| pytest | ≥ 7.0 | API tests |
| httpx | ≥ 0.27 | Test client |
| python-multipart | ≥ 0.0.9 | Form parsing |
