# ZeroTrace

> An intelligent, AI-powered carbon footprint tracker that helps individuals monitor and reduce their environmental impact — no spreadsheets, no paid APIs, no cloud dependency.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![Ollama](https://img.shields.io/badge/Ollama-qwen2.5:14b-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

ZeroTrace is a web-based application that tracks carbon emissions across three everyday activities — **device usage**, **electricity consumption**, and **transportation** — and uses a locally-hosted large language model (Ollama + qwen2.5:14b) to generate personalised, actionable reduction insights.

Unlike cloud-dependent tools, ZeroTrace runs entirely on your machine. No API keys. No subscription. No data leaves your device.

---

## Features

| Feature | Description |
|---|---|
| **Device Carbon Analyser** | Calculate CO₂ emissions from any device given wattage and daily usage hours |
| **Electricity Bill Scanner** | Upload a bill image; AI extracts units consumed via OCR + LLM parsing |
| **Transport Calculator** | Estimate emissions by mode (car, metro, bus, train, flight) and distance |
| **AI Recommendations** | qwen2.5:14b generates specific, device-aware tips after each calculation |
| **Eco Alternatives** | Transport module suggests a lower-carbon alternative for every trip |
| **Dashboard & History** | Visualise trends across all three modules over time |
| **Cross-module Insights** | `/api/ai/insights` endpoint delivers an overall carbon audit from all history |
| **Fully offline AI** | All inference runs locally via Ollama — GPU-accelerated on supported hardware |

---

## Tech Stack

**Backend**
- Python 3.10+, Flask 3.x
- EasyOCR (bill text extraction)
- OpenCV, NumPy (image preprocessing)
- Ollama Python client (`ollama`)

**AI / ML**
- [Ollama](https://ollama.com) — local LLM runtime
- `qwen2.5:14b` — primary model for recommendations, OCR parsing, eco alternatives

**Frontend**
- HTML5, CSS3, vanilla JavaScript
- Chart.js (dashboard visualisations)

**Storage**
- JSON flat-file history (per module, append-only)

---

## Project Structure

```
ZeroTrace/
├── app.py                  # Flask routes, AI integration, history I/O
├── device.py               # Device emission calculations + rule-based tips
├── transport.py            # Transport emission factors + calculator
├── ocr.py                  # OCR extraction, receipt parsing (legacy)
├── static/
│   ├── index.html
│   ├── dashboard.html
│   ├── device_carbon_analyzer.html
│   ├── transport_emissions.html
│   ├── bill.html
│   ├── device_emissions_history.json
│   ├── transport_emissions.json
│   └── electricity_history.json
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed and running
- NVIDIA GPU recommended (RTX series with 8GB+ VRAM for full GPU inference)

### 1. Clone the repository

```bash
git clone https://github.com/Bhavishya-Chaturvedi/ZeroTrace.git
cd ZeroTrace
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Ollama and pull the model

```bash
# Install Ollama from https://ollama.com/download, then:
ollama pull qwen2.5:14b
```

### 4. Enable GPU acceleration (recommended)

Set the following environment variable before starting Ollama so the full model loads into VRAM:

```bash
# Windows
set OLLAMA_GPU_LAYERS=40

# Linux / macOS
export OLLAMA_GPU_LAYERS=40
```

Then start the Ollama server:

```bash
ollama serve
```

> `40` layers loads the entire `qwen2.5:14b` model into ~9GB VRAM. Lower to `35` if you encounter out-of-memory errors.

### 5. Run the application

```bash
python app.py
```

Open your browser at `http://localhost:10000`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/device/calculate` | Analyse device emissions; returns `ai_recommendations[]` |
| `POST` | `/transport/calculate` | Calculate trip emissions; returns `ai_alternative{}` |
| `POST` | `/electricity/upload` | Upload bill image; AI extracts units, falls back to crop |
| `POST` | `/electricity/save_manual` | Save manually entered electricity units |
| `GET` | `/device/history` | Retrieve device emission history |
| `GET` | `/transport/history` | Retrieve transport emission history |
| `GET` | `/electricity/history` | Retrieve electricity bill history |
| `POST` | `/api/ai/insights` | Cross-module AI carbon audit summary |
| `GET` | `/api/ai/status` | Check Ollama availability and active model |

---

## AI Integration

ZeroTrace uses `qwen2.5:14b` (via Ollama) for three tasks:

**1. Electricity bill parsing** — OCR text from any bill layout is sent to the LLM, which extracts the units consumed field contextually. Falls back to a pixel-crop method if AI is unavailable.

**2. Device recommendations** — After rule-based calculation, the top emitting devices are sent to the model, which returns three specific, device-named reduction tips.

**3. Transport eco alternatives** — For every trip, the model suggests the single best lower-carbon alternative given the distance and India-specific transit options.

All AI calls are non-blocking (thread pool) and degrade gracefully — the app works fully without Ollama, returning rule-based results instead.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_GPU_LAYERS` | `0` | Number of model layers to offload to GPU. Set to `40` for full VRAM loading on RTX GPUs |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "add: your feature description"`
4. Push and open a pull request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Ollama](https://ollama.com) for making local LLM inference accessible
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for the OCR pipeline
- [Qwen team](https://github.com/QwenLM/Qwen2.5) for the open-weight model
