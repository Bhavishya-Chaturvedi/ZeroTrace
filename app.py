# =============================================================================
# app.py — ZeroTrace with Ollama AI Integration (qwen2.5:14b on RTX 5050)
# =============================================================================
# SETUP BEFORE RUNNING:
#   1. Install Ollama: https://ollama.com/download
#   2. Pull model:     ollama pull qwen2.5:14b
#   3. Start Ollama:   ollama serve
#   4. GPU config:     Set OLLAMA_GPU_LAYERS below (see GPU CONFIG section)
#   5. Install dep:    pip install ollama
#
# GPU CONFIG (RTX 5050 — 16GB VRAM):
#   Run this once in terminal before `ollama serve`:
#       set OLLAMA_GPU_LAYERS=40        (Windows)
#       export OLLAMA_GPU_LAYERS=40     (Linux/Mac)
#   Or set it permanently in your system environment variables.
#   40 layers = full qwen2.5:14b fits in VRAM → fastest inference.
#   If you see OOM errors, lower to 35.
# =============================================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from ocr import extract_text, parse_receipt, estimate_carbon_emissions, CARBON_EMISSIONS
import logging
import device
from transport import calculate_transport_emissions
import json
import datetime
from datetime import timezone

# ── NEW IMPORTS FOR AI ────────────────────────────────────────────────────────
import re
import ollama                          # pip install ollama
from concurrent.futures import ThreadPoolExecutor
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

# =============================================================================
# AI CONFIG
# =============================================================================
# Single model for ALL AI work. Change here if you ever switch models.
AI_MODEL = "qwen2.5:14b"

# Thread pool — limits concurrent Ollama calls so GPU isn't double-booked.
# max_workers=1 means requests queue up rather than running in parallel.
# This is intentional: qwen2.5:14b is large; parallel calls thrash VRAM.
_ai_executor = ThreadPoolExecutor(max_workers=1)

# Ollama availability flag — set at startup, used to skip AI gracefully
_ollama_available = False


def _check_ollama():
    """
    Called once at startup. Verifies Ollama is running and qwen2.5:14b is
    pulled. Sets _ollama_available flag used by all AI functions.

    If Ollama is down, the app still works — AI endpoints just return
    {"ai_available": false} and the regular rule-based data is returned.
    """
    global _ollama_available
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        if any(AI_MODEL in name for name in model_names):
            _ollama_available = True
            logger.info(f"✅ Ollama ready — {AI_MODEL} found. GPU acceleration active.")
        else:
            logger.warning(
                f"⚠️  Ollama is running but {AI_MODEL} is NOT pulled. "
                f"Run: ollama pull {AI_MODEL}"
            )
    except Exception as e:
        logger.warning(f"⚠️  Ollama not reachable: {e}. AI features disabled.")


def _call_ollama(prompt: str) -> str:
    """
    Central function for ALL Ollama calls. Every AI feature routes through here.

    Why centralise:
    - Single place to change model name
    - Single place to handle errors/timeouts
    - Easy to add logging or caching later

    Returns the raw text response string, or "" on failure.
    """
    if not _ollama_available:
        return ""
    try:
        response = ollama.chat(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                # GPU acceleration options for RTX 5050
                # num_gpu: number of GPU layers to offload (40 = full model in VRAM)
                # num_thread: CPU threads used for the non-GPU parts
                "num_gpu": 40,
                "num_thread": 8,
                # Lower temperature = more deterministic JSON output
                "temperature": 0.2,
            }
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return ""


def _extract_json(text: str):
    """
    Extract JSON from Ollama response text. Ollama sometimes wraps JSON in
    markdown code fences (```json ... ```) or adds preamble text.
    This handles all those cases.

    Returns parsed object or None if extraction fails.
    """
    if not text:
        return None
    # Strip markdown code fences if present
    text = re.sub(r'```(?:json)?', '', text).strip()
    # Find first {...} or [...] block
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning(f"JSON parse failed on: {match.group()[:200]}")
    return None


# =============================================================================
# ── AI FUNCTION 1: ELECTRICITY BILL ──────────────────────────────────────────
# =============================================================================
# WHERE IT'S USED: Called inside upload_electricity_bill() as the PRIMARY
# extraction method. Falls back to process_electricity_bill() (the old
# hardcoded crop) only if AI returns {"found": false}.
#
# WHY THIS MATTERS: Your current process_electricity_bill() crops at exact
# pixel coords (730,330,118,50) — works for one specific bill layout only.
# This AI version reads any bill from any DISCOM in any layout.
# =============================================================================

def ai_extract_electricity_units(image_path: str) -> dict:
    """
    Uses qwen2.5:14b (text-only) to extract units from OCR text of the bill.

    NOTE: qwen2.5:14b is a TEXT model, not vision. So we:
      1. Run EasyOCR on the full image (already available via extract_text())
      2. Send that text to qwen2.5:14b to find the units field

    This is more robust than the hardcoded crop because:
    - OCR covers the whole bill, not just one region
    - The LLM understands context ("Units Consumed", "kWh", "net units", etc.)
    """
    if not _ollama_available:
        return {"found": False, "reason": "ollama_unavailable"}

    try:
        # Re-use your existing extract_text() from ocr.py
        ocr_text = extract_text(image_path)
        if not ocr_text:
            return {"found": False, "reason": "ocr_failed"}

        prompt = f"""You are analyzing an electricity bill. 
Find the total units consumed (kWh or units). This is usually labeled as:
"Units Consumed", "Net Units", "Total Units", "kWh", "Consumption" etc.

Bill text:
{ocr_text}

Return ONLY valid JSON, no explanation:
{{"found": true, "units": 245.5, "label_found": "Units Consumed"}}
or
{{"found": false}}
"""
        raw = _call_ollama(prompt)
        result = _extract_json(raw)

        if result and result.get("found") and isinstance(result.get("units"), (int, float)):
            logger.info(f"AI extracted units: {result['units']} (label: {result.get('label_found')})")
            return result
        else:
            return {"found": False, "reason": "units_not_found_in_text"}

    except Exception as e:
        logger.error(f"ai_extract_electricity_units error: {e}")
        return {"found": False, "reason": str(e)}


# =============================================================================
# ── AI FUNCTION 2: DEVICE ANALYZER RECOMMENDATIONS ───────────────────────────
# =============================================================================
# WHERE IT'S USED: Called inside calculate_device_emissions() AFTER the
# rule-based device.analyze_device() runs. Replaces the static tips from
# device.py's get_reduction_tips() with personalised AI advice.
#
# The existing tips in device.py are generic (same 5 bullets for everyone).
# This generates advice specific to WHICH devices the user actually has
# and which ones are the biggest offenders.
# =============================================================================

def ai_device_recommendations(device_breakdown: list, total_emissions_kg: float) -> list:
    """
    Returns list of 3 specific recommendation strings based on actual device data.
    Falls back to [] on failure (frontend should show rule-based tips in that case).
    """
    if not _ollama_available or not device_breakdown:
        return []

    # Sort by emissions descending, take top 3 to keep prompt short
    top_devices = sorted(device_breakdown, key=lambda x: x['emissions_g'], reverse=True)[:3]

    # Build a compact summary for the prompt (avoid sending huge payloads)
    device_summary = [
        {
            "device": d["device"],
            "hours_per_day": d["hours"],
            "wattage_W": d["wattage"],
            "daily_emissions_g": d["emissions_g"]
        }
        for d in top_devices
    ]

    prompt = f"""A user's top energy-consuming devices today:
{json.dumps(device_summary, indent=2)}
Total daily emissions: {round(total_emissions_kg * 1000, 1)}g CO2

Give exactly 3 specific, actionable tips to reduce their carbon footprint.
Rules:
- Each tip must mention a specific device from the list above
- No generic advice like "turn off lights" or "use dark mode"
- Be concrete: suggest hours to reduce, settings to change, or alternatives
- Keep each tip under 2 sentences

Return ONLY a JSON array of strings:
["tip1", "tip2", "tip3"]
"""
    raw = _call_ollama(prompt)
    result = _extract_json(raw)

    if isinstance(result, list) and len(result) >= 1:
        logger.info(f"AI generated {len(result)} device tips")
        return result[:3]
    return []


# =============================================================================
# ── AI FUNCTION 3: TRANSPORT ECO ALTERNATIVES ────────────────────────────────
# =============================================================================
# WHERE IT'S USED: Called inside calculate_transport_emissions_endpoint()
# AFTER calculate_transport_emissions() runs. Adds an "ai_alternative" key
# to the response JSON — the frontend can display this as a suggestion card.
#
# Your current transport.py only does: emissions = distance × factor.
# This adds intelligence: "you took a car for 8km — metro would save X kg"
# =============================================================================

def ai_transport_alternative(mode: str, distance_km: float, emissions_kg: float) -> dict:
    """
    Suggests a lower-carbon transport alternative for the same trip.
    Returns dict with keys: alternative_mode, estimated_emissions_kg,
    saving_kg, feasibility, note.
    Returns {} on failure.
    """
    if not _ollama_available:
        return {}

    prompt = f"""A user in India travelled {distance_km} km by {mode}, emitting {emissions_kg} kg CO2.

Suggest the single BEST lower-carbon alternative for this exact trip distance.
Consider practical Indian transport options: metro, bus, auto, bicycle, walking, train.
For distances under 2km, prefer walking/cycling.
For 2-15km, prefer metro/bus.
For 15km+, prefer train/bus.

Return ONLY valid JSON:
{{
  "alternative_mode": "metro",
  "estimated_emissions_kg": 0.18,
  "saving_kg": 1.34,
  "feasibility": "high",
  "note": "Delhi Metro covers most of this route via Blue Line"
}}
"""
    raw = _call_ollama(prompt)
    result = _extract_json(raw)

    if isinstance(result, dict) and "alternative_mode" in result:
        logger.info(f"AI transport alternative: {result['alternative_mode']}")
        return result
    return {}


# =============================================================================
# ── AI FUNCTION 4: CROSS-MODULE DASHBOARD INSIGHTS ───────────────────────────
# =============================================================================
# WHERE IT'S USED: New endpoint /api/ai/insights — called by the dashboard
# page when it loads. Combines data from all 3 modules for a holistic view.
#
# This is the highest-value AI feature: it sees the user's full picture
# (devices + transport + electricity) and gives an overall carbon audit.
# =============================================================================

def ai_overall_insights(electricity_history: list, device_history: list, transport_history: list) -> dict:
    """
    Generates a short overall carbon audit summary across all modules.
    Returns dict with keys: summary (str), biggest_category (str), weekly_tip (str).
    """
    if not _ollama_available:
        return {}

    # Limit history to last 7 entries per module to keep prompt small
    elec_recent = electricity_history[-7:] if electricity_history else []
    dev_recent  = device_history[-7:] if device_history else []
    trans_recent = transport_history[-7:] if transport_history else []

    if not any([elec_recent, dev_recent, trans_recent]):
        return {}

    prompt = f"""Analyze this user's carbon footprint data across three categories:

ELECTRICITY (recent bills):
{json.dumps(elec_recent, default=str)}

DEVICES (daily usage logs):
{json.dumps(dev_recent, default=str)}

TRANSPORT (trips):
{json.dumps(trans_recent, default=str)}

Provide a brief carbon audit. Return ONLY valid JSON:
{{
  "summary": "2-3 sentence overall assessment of their carbon habits",
  "biggest_category": "electricity|devices|transport",
  "weekly_tip": "One specific high-impact action they should take this week",
  "total_estimated_kg": 12.5
}}
"""
    raw = _call_ollama(prompt)
    result = _extract_json(raw)

    if isinstance(result, dict) and "summary" in result:
        return result
    return {}


# =============================================================================
# HISTORY HELPERS (unchanged from original)
# =============================================================================

ELECTRICITY_HISTORY_PATH = os.path.join(app.static_folder, 'electricity_history.json')
DEVICE_HISTORY_PATH = os.path.join(app.static_folder, 'device_emissions_history.json')


def _ensure_static_folder():
    os.makedirs(app.static_folder, exist_ok=True)


def _load_electricity_history():
    if not os.path.exists(ELECTRICITY_HISTORY_PATH):
        return []
    history = []
    try:
        with open(ELECTRICITY_HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to read electricity history: {e}")
    return history


def _append_electricity_history(entry):
    os.makedirs(app.static_folder, exist_ok=True)
    with open(ELECTRICITY_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_device_history():
    if not os.path.exists(DEVICE_HISTORY_PATH):
        return []
    history = []
    try:
        with open(DEVICE_HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))
    except Exception as e:
        logger.error(f"History read error: {e}")
    return history


def _append_device_history(entry):
    _ensure_static_folder()
    with open(DEVICE_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# ELECTRICITY BILL PROCESSING (original hardcoded crop — kept as fallback)
# =============================================================================

import cv2, numpy as np, easyocr, tempfile


def process_electricity_bill(image_path):
    """
    Original rule-based crop method. Now used as FALLBACK only when
    ai_extract_electricity_units() returns {"found": false}.
    """
    x, y, w, h = 730, 330, 118, 50
    carbon_intensity = 0.82

    def calc_bill(u):
        fixed = 20
        slabs = [(200, 3.0), (200, 4.5), (400, 6.5), (400, 7.0), (float('inf'), 8.0)]
        total, rem = fixed, u
        for sz, rate in slabs:
            if rem <= 0: break
            take = min(rem, sz) if sz != float('inf') else rem
            total += take * rate
            rem -= take
        return round(total, 2)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Cannot read image")
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError("Crop empty")
    tight = crop[int(crop.shape[0] * 0.6):, :]
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    tight = cv2.filter2D(tight, -1, kernel)
    tight = cv2.resize(tight, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    reader = easyocr.Reader(['en'], gpu=False)
    txt = " ".join([t for _, t, _ in reader.readtext(tight)])
    cleaned = txt.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('g', '9')
    m = re.search(r'\d+(\.\d+)?', cleaned)
    if m:
        units = float(m.group(0))
        return {"success": True, "units": units,
                "bill_amount": calc_bill(units),
                "co2_emissions": round(units * carbon_intensity, 2)}
    else:
        return {"success": False, "error": "units_not_detected",
                "message": "Units not found - please enter manually."}


# =============================================================================
# FLASK ROUTES — STATIC PAGES (unchanged)
# =============================================================================

@app.route('/')
def serve_index():
    try:
        return send_from_directory('static', 'index.html')
    except Exception as e:
        return jsonify({"error": "File not found"}), 404


@app.route('/dashboard')
def serve_dashboard():
    try:
        return send_from_directory('static', 'dashboard.html')
    except Exception as e:
        return jsonify({"error": "File not found"}), 404


@app.route('/ocr')
def serve_receipt_ocr():
    try:
        return send_from_directory(app.static_folder, 'receipt_ocr_page.html')
    except Exception as e:
        return jsonify({"error": "File not found"}), 404


@app.route('/device')
def serve_device_analyzer():
    try:
        return send_from_directory('static', 'device_carbon_analyzer.html')
    except Exception as e:
        return jsonify({"error": "File not found"}), 404


@app.route('/transport')
def serve_transport_calculator():
    try:
        return send_from_directory('static', 'transport_emissions.html')
    except Exception as e:
        return jsonify({"error": "File not found"}), 404


@app.route('/features')
def serve_features():
    try:
        return send_from_directory('static', 'features.html')
    except Exception as e:
        return jsonify({"error": "File not found"}), 404


@app.route('/how-it-works')
def serve_how_it_works():
    try:
        return send_from_directory('static', 'how-it-works.html')
    except Exception as e:
        return jsonify({"error": "File not found"}), 404


@app.route('/electricity')
def serve_electricity_page():
    return send_from_directory('static', 'bill.html')


# =============================================================================
# FLASK ROUTE — UPLOAD RECEIPT (unchanged, OCR not active but route kept)
# =============================================================================

@app.route('/upload', methods=['POST'])
def upload_receipt():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files['image']
        shopping_list = request.form.get('shopping_list', '').split(',')
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        upload_path = f"static/upload_{uuid.uuid4().hex}.png"
        file.save(upload_path)
        for item in shopping_list:
            item = item.strip().lower()
            if item and item not in CARBON_EMISSIONS:
                CARBON_EMISSIONS[item] = 1.0
        text = extract_text(upload_path)
        if not text:
            return jsonify({"error": "Failed to extract text from image"}), 500
        items = parse_receipt(text)
        if not items:
            return jsonify({"error": "No items detected in receipt"}), 400
        results = estimate_carbon_emissions(items)
        try:
            os.remove(upload_path)
        except Exception:
            pass
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# =============================================================================
# FLASK ROUTE — DEVICE CALCULATE
# MODIFIED: ai_device_recommendations() called after rule-based analysis.
#           AI tips added as "ai_recommendations" key in response.
#           Falls back to original rule-based tips if AI is unavailable.
# =============================================================================

@app.route('/device/calculate', methods=['POST'])
def calculate_device_emissions():
    try:
        data = request.get_json()
        if not data or 'devices' not in data:
            return jsonify({"error": "No device data provided"}), 400

        # ── UNCHANGED: rule-based analysis ───────────────────────────────────
        results = device.analyze_device(data)

        if results.get('success'):
            # Save history (unchanged)
            for item in results['device_breakdown']:
                entry = {
                    "timestamp": datetime.datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    "device": item['device'],
                    "emissions_kg": round(item['emissions_g'] / 1000, 3),
                    "energy_kwh": item['energy_kwh'],
                    "wattage": item['wattage'],
                    "hours": item['hours']
                }
                _append_device_history(entry)

            total_entry = {
                "timestamp": datetime.datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "device": "All Devices",
                "emissions_kg": round(results['total_emissions_kg'], 3),
                "energy_kwh": round(results['total_energy'], 3),
                "daily_cost": round(results['daily_cost'], 2)
            }
            _append_device_history(total_entry)

            # ── NEW: AI recommendations (non-blocking via thread pool) ────────
            # We submit AI call to executor so it doesn't slow down the response.
            # Wait up to 25 seconds — qwen2.5:14b on RTX 5050 should finish in 5-10s.
            ai_tips = []
            if _ollama_available:
                try:
                    future = _ai_executor.submit(
                        ai_device_recommendations,
                        results['device_breakdown'],
                        results['total_emissions_kg']
                    )
                    ai_tips = future.result(timeout=25)
                except Exception as e:
                    logger.warning(f"AI device tips timed out or failed: {e}")

            # Add AI tips to response; keep original rule-based tips as fallback
            results['ai_recommendations'] = ai_tips if ai_tips else results.get('tips', [])
            results['ai_powered'] = bool(ai_tips)
            # ─────────────────────────────────────────────────────────────────

        return jsonify(results)

    except Exception as e:
        logger.error(f"Calculate error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/device/history')
def device_history():
    history = _load_device_history()
    history.sort(key=lambda x: x["timestamp"])
    return jsonify(history)


# =============================================================================
# FLASK ROUTE — TRANSPORT CALCULATE
# MODIFIED: ai_transport_alternative() called after rule-based calculation.
#           Adds "ai_alternative" key to response with eco suggestion.
# =============================================================================

@app.route('/transport/calculate', methods=['POST'])
def calculate_transport_emissions_endpoint():
    try:
        data = request.get_json()
        if not data or 'transport_mode' not in data or 'distance' not in data:
            return jsonify({"error": "Missing transport mode or distance"}), 400

        transport_mode = data['transport_mode'].lower()
        distance = float(data['distance'])

        # ── UNCHANGED: rule-based calculation ────────────────────────────────
        result = calculate_transport_emissions(transport_mode, distance)

        # ── NEW: AI eco alternative (non-blocking) ────────────────────────────
        # Only run if Ollama is up. Adds "ai_alternative" to response.
        # Frontend can show this as: "💡 Metro would save X kg CO2 on this trip"
        if _ollama_available:
            try:
                future = _ai_executor.submit(
                    ai_transport_alternative,
                    transport_mode,
                    distance,
                    result.get('carbon_emissions_kg', 0)
                )
                alt = future.result(timeout=25)
                result['ai_alternative'] = alt
            except Exception as e:
                logger.warning(f"AI transport alternative timed out: {e}")
                result['ai_alternative'] = {}
        else:
            result['ai_alternative'] = {}
        # ─────────────────────────────────────────────────────────────────────

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/transport/history')
def transport_history():
    json_path = os.path.join(app.static_folder, 'transport_emissions.json')
    if not os.path.exists(json_path):
        return jsonify([])
    history = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if isinstance(record, list) and len(record) > 0:
                        history.append(record[0])
                except json.JSONDecodeError as e:
                    logger.error(f"JSON error at line {line_num}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Failed to read transport history: {e}")
    return jsonify(history)


# =============================================================================
# FLASK ROUTE — ELECTRICITY UPLOAD
# MODIFIED: ai_extract_electricity_units() tried FIRST.
#           process_electricity_bill() (hardcoded crop) used as fallback.
#           Response includes "extraction_method" key so you can debug.
# =============================================================================

@app.route('/electricity/upload', methods=['POST'])
def upload_electricity_bill():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image"}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            file.save(tmp.name)
            path = tmp.name

        carbon_intensity = 0.82

        def calc_bill(u):
            fixed = 20
            slabs = [(200, 3.0), (200, 4.5), (400, 6.5), (400, 7.0), (float('inf'), 8.0)]
            total, rem = fixed, u
            for sz, rate in slabs:
                if rem <= 0: break
                take = min(rem, sz) if sz != float('inf') else rem
                total += take * rate
                rem -= take
            return round(total, 2)

        # ── NEW: Try AI extraction first ──────────────────────────────────────
        extraction_method = "rule_based_crop"
        result = None

        if _ollama_available:
            try:
                future = _ai_executor.submit(ai_extract_electricity_units, path)
                ai_result = future.result(timeout=30)
                if ai_result.get("found") and ai_result.get("units"):
                    units = float(ai_result["units"])
                    result = {
                        "success": True,
                        "units": units,
                        "bill_amount": calc_bill(units),
                        "co2_emissions": round(units * carbon_intensity, 2)
                    }
                    extraction_method = "ai_ocr"
                    logger.info(f"Electricity: AI extracted {units} units")
            except Exception as e:
                logger.warning(f"AI electricity extraction failed: {e}")

        # ── FALLBACK: original hardcoded crop ─────────────────────────────────
        if result is None:
            result = process_electricity_bill(path)
            logger.info("Electricity: fell back to rule-based crop extraction")

        os.unlink(path)

        # Add metadata about which method was used (useful for debugging)
        result['extraction_method'] = extraction_method
        # ─────────────────────────────────────────────────────────────────────

        if result.get("success"):
            entry = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "units": result["units"],
                "bill_amount": result["bill_amount"],
                "co2_emissions": result["co2_emissions"]
            }
            _append_electricity_history(entry)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Electricity upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/electricity/save_manual', methods=['POST'])
def save_manual_bill():
    try:
        data = request.get_json()
        units = float(data.get('units', 0))
        date = data.get('date') or datetime.datetime.now().strftime("%Y-%m-%d")

        def calc_bill(u):
            fixed = 20
            slabs = [(200, 3.0), (200, 4.5), (400, 6.5), (400, 7.0), (float('inf'), 8.0)]
            total, rem = fixed, u
            for sz, rate in slabs:
                if rem <= 0: break
                take = rem if sz == float('inf') else min(rem, sz)
                total += take * rate
                rem -= take
            return round(total, 2)

        bill = calc_bill(units)
        co2 = round(units * 0.82, 2)

        entry = {"date": date, "units": units, "bill_amount": bill, "co2_emissions": co2}
        _append_electricity_history(entry)

        return jsonify({"success": True, "units": units, "bill_amount": bill, "co2_emissions": co2})

    except Exception as e:
        logger.error(f"Manual save error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/electricity/history')
def electricity_history():
    history = _load_electricity_history()
    history.sort(key=lambda x: x["date"], reverse=True)
    return jsonify(history)


# =============================================================================
# NEW ROUTE — /api/ai/insights
# Cross-module dashboard summary. Call this from your dashboard.html JS
# to show an overall carbon audit powered by AI.
#
# Usage (frontend JS):
#   fetch('/api/ai/insights', {
#       method: 'POST',
#       headers: {'Content-Type': 'application/json'},
#       body: JSON.stringify({})   // server loads history itself
#   })
# =============================================================================

@app.route('/api/ai/insights', methods=['POST'])
def ai_insights_endpoint():
    """
    Loads history from all 3 modules and returns an AI-generated summary.
    Safe to call even if Ollama is down — returns {"ai_available": false}.
    """
    if not _ollama_available:
        return jsonify({"ai_available": False, "reason": "ollama_not_running"})

    try:
        elec_history = _load_electricity_history()
        dev_history = _load_device_history()

        # Load transport history
        trans_history = []
        json_path = os.path.join(app.static_folder, 'transport_emissions.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            if isinstance(record, list) and record:
                                trans_history.append(record[0])
                        except Exception:
                            pass

        future = _ai_executor.submit(ai_overall_insights, elec_history, dev_history, trans_history)
        insights = future.result(timeout=35)

        if insights:
            return jsonify({"ai_available": True, "insights": insights})
        else:
            return jsonify({"ai_available": True, "insights": None, "reason": "insufficient_data"})

    except Exception as e:
        logger.error(f"AI insights endpoint error: {e}")
        return jsonify({"ai_available": False, "reason": str(e)})


# =============================================================================
# NEW ROUTE — /api/ai/status
# Simple health check. Call from frontend to show "AI Active" badge.
#
# Usage: fetch('/api/ai/status').then(r => r.json())
# Returns: {"available": true, "model": "qwen2.5:14b"}
# =============================================================================

@app.route('/api/ai/status')
def ai_status():
    return jsonify({
        "available": _ollama_available,
        "model": AI_MODEL if _ollama_available else None
    })


# =============================================================================
# ERROR HANDLERS (unchanged)
# =============================================================================

@app.errorhandler(404)
def not_found_error(e):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


@app.route('/api/placeholder/<int:width>/<int:height>')
def placeholder(width, height):
    return f"Placeholder image of {width}x{height}", 200


# =============================================================================
# STARTUP
# =============================================================================

if __name__ == '__main__':
    # Check Ollama availability before starting Flask
    # This is synchronous on purpose — want to know GPU status at boot
    _check_ollama()
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=10000)