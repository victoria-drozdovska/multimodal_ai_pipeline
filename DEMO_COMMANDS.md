# Demo Commands (End-to-End)

Run commands in order.

## 1) Activate the Environment and Start API server
```bash
cd /path/to/multimodal_ai_pipeline
source .venv/bin/activate
python app.py
```
## Check Backend
```bash
curl -sS http://127.0.0.1:5001/health | python3 -m json.tool
```

## 2) Open UI
```bash
open http://127.0.0.1:5001/ui
```

## Proceed in UI