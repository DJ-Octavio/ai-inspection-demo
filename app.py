"""
AI 智检 · Flask API 推理服务（接入 PatchCore 真实模型）
启动：cd C:\\Users\\rui.ma5\\aizj && python app.py
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2
import torch

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==================== CONFIG ====================
MODEL_PATH = Path("./weights/model.ckpt")
THRESHOLD_OK = 0.3      # anomaly_score < this = OK
THRESHOLD_WARN = 0.6    # anomaly_score between OK and WARN = WARN (human confirm)
                        # anomaly_score > WARN = NG

DEFECT_NAMES = {
    "good": "no defect",
    "anomaly": "anomaly detected",
}

# ==================== MODEL LOADING ====================
model = None          # lightning module (kept for reference)
raw_model = None      # raw torch PatchcoreModel — gives un-normalized separable scores
transform = None
OK_THR = 3.0          # raw score < OK_THR  -> OK   (calibrated at load)
NG_THR = 3.3          # raw score > NG_THR  -> NG   (calibrated at load)
CALIB_DIR = Path("./raw_images_training/good")  # good samples for threshold calibration


def _calibrate_thresholds():
    """Run good training images through raw model to calibrate OK/NG thresholds.

    Root cause of the all-1.0 bug: anomalib's lightning post-processor min-max
    normalizes with a degenerate validation set, collapsing every score to 1.0.
    The raw torch model (model.model) gives clean separable scores, so we
    calibrate thresholds directly from the good samples here.
    """
    global OK_THR, NG_THR
    if raw_model is None or not CALIB_DIR.exists():
        print(f"[WARN] Calibration dir missing, using default thresholds OK<{OK_THR} NG>{NG_THR}")
        return

    imgs = [f for f in CALIB_DIR.glob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    if len(imgs) < 5:
        print(f"[WARN] Too few calibration images ({len(imgs)}), using defaults")
        return

    scores = []
    for p in imgs[:80]:
        try:
            img = Image.open(p).convert('RGB')
            with torch.no_grad():
                scores.append(float(raw_model(transform(img).unsqueeze(0)).pred_score.item()))
        except Exception:
            pass

    if len(scores) < 5:
        return

    scores = np.array(scores)
    # OK threshold: most good images sit below this (p85)
    OK_THR = float(np.percentile(scores, 85))
    # NG threshold: clearly above the worst good image (max * margin)
    NG_THR = float(max(np.percentile(scores, 99), scores.max() * 1.03))
    if NG_THR <= OK_THR:
        NG_THR = OK_THR + 0.3
    print(f"[OK] Calibrated thresholds from {len(scores)} good images:")
    print(f"     good: mean={scores.mean():.3f} max={scores.max():.3f}")
    print(f"     OK_THR={OK_THR:.3f}  NG_THR={NG_THR:.3f}")


def load_model():
    """Load PatchCore model from checkpoint"""
    global model, raw_model, transform
    if not MODEL_PATH.exists():
        print(f"[WARN] Model not found: {MODEL_PATH}")
        print("       Running in simulation mode")
        return False

    try:
        import timm
        # Patch timm to use local weights (bypass HuggingFace)
        local_weights = os.path.join(os.path.dirname(__file__), 'weights', 'wide_resnet50_2-95faca4d.pth')
        if Path(local_weights).exists():
            _orig_create = timm.create_model
            def _patched_create(name, pretrained=False, **kw):
                if name == 'wide_resnet50_2' and pretrained:
                    m = _orig_create(name, pretrained=False, **kw)
                    sd = torch.load(local_weights, map_location='cpu', weights_only=False)
                    m.load_state_dict(sd, strict=False)
                    return m
                return _orig_create(name, pretrained=pretrained, **kw)
            timm.create_model = _patched_create

        from anomalib.models import Patchcore

        # Load model from checkpoint (weights_only=False for anomalib compatibility)
        model = Patchcore.load_from_checkpoint(str(MODEL_PATH), map_location='cpu', weights_only=False)
        model.eval()
        # Use the RAW torch model — bypasses the broken lightning normalization
        raw_model = model.model
        raw_model.eval()

        # Setup transform (same as training)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("[OK] PatchCore model loaded successfully")
        print(f"     Model path: {MODEL_PATH}")

        # Calibrate thresholds from good samples
        _calibrate_thresholds()
        return True
    except Exception as e:
        print(f"[WARN] Model load failed: {e}")
        print("       Running in simulation mode")
        model = None
        raw_model = None
        return False


def _extract_bbox_and_type(image_pil, anomaly_map):
    """Extract defect bounding box from anomaly map + heuristic defect type.

    anomaly_map: torch tensor [1,1,H,W] or [1,H,W] of patch distances.
    Returns (bbox_normalized [x,y,w,h] or None, type_key, type_cn).
    """
    try:
        amap = anomaly_map.squeeze().detach().cpu().numpy().astype(np.float32)  # HxW
        H, W = amap.shape
        # Hotspot = top 4% anomalous pixels
        thr = np.percentile(amap, 96)
        mask = (amap >= thr).astype(np.uint8)
        # Keep the largest connected blob for a clean box
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None, "scratch", "外观划痕"
        # stats[0] is background; pick largest non-background by area
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = stats[idx, 0], stats[idx, 1], stats[idx, 2], stats[idx, 3]
        # pad box a little
        pad = max(2, int(0.04 * max(W, H)))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        bbox = [round(x0 / W, 4), round(y0 / H, 4), round((x1 - x0) / W, 4), round((y1 - y0) / H, 4)]

        # ===== Defect type heuristic on the original image crop =====
        ow, oh = image_pil.size
        cx, cy = int(bbox[0] * ow), int(bbox[1] * oh)
        cw, ch = max(12, int(bbox[2] * ow)), max(12, int(bbox[3] * oh))
        crop = np.array(image_pil.crop((cx, cy, min(ow, cx + cw), min(oh, cy + ch))).convert('RGB'))
        if crop.size == 0:
            return bbox, "scratch", "外观划痕"
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        full_gray = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
        edge_density = float(cv2.Canny(gray, 50, 150).mean()) / 255.0
        brightness = float(gray.mean()) / 255.0
        contrast_ratio = float(gray.std()) / (float(full_gray.std()) + 1e-6)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hue_std = float(hsv[:, :, 0].std())

        if brightness < 0.32:
            return bbox, "dent", "凹坑/压伤"
        if edge_density > 0.07 or contrast_ratio > 1.25:
            return bbox, "scratch", "外观划痕"
        if hue_std > 28:
            return bbox, "color", "色差/偏色"
        return bbox, "scratch", "外观划痕"
    except Exception:
        return None, "scratch", "外观划痕"


def predict_image(image_pil):
    """Run PatchCore inference on a PIL image (uses raw model + calibrated thresholds)"""
    global raw_model, transform

    if raw_model is None or transform is None:
        return simulate_predict(image_pil)

    try:
        img_tensor = transform(image_pil).unsqueeze(0)  # [1, 3, 256, 256]

        # Inference via RAW torch model (un-normalized, separable scores)
        with torch.no_grad():
            output = raw_model(img_tensor)
        raw_score = float(output.pred_score.item())

        # Normalized 0-1 anomaly score for display: OK_THR -> 0, NG_THR -> 1
        span = max(NG_THR - OK_THR, 1e-6)
        norm_anomaly = max(0.0, min(1.0, (raw_score - OK_THR) / span))

        bbox = None
        if raw_score < OK_THR:
            verdict = "ok"; defect_type = "no defect"; defect_key = "good"; type_cn = "无缺陷"
        else:
            # localize defect + guess type
            bbox, defect_key, type_cn = _extract_bbox_and_type(image_pil, output.anomaly_map)
            if raw_score < NG_THR:
                verdict = "warn"; defect_type = type_cn + "（待确认）"
            else:
                verdict = "ng"; defect_type = type_cn

        confidence = 1.0 - norm_anomaly if verdict == "ok" else norm_anomaly

        return {
            "type": defect_type,
            "type_key": defect_key if raw_score >= OK_THR else "good",
            "type_cn": type_cn if raw_score >= OK_THR else "无缺陷",
            "confidence": round(confidence, 3),
            "anomaly_score": round(norm_anomaly, 3),
            "raw_score": round(raw_score, 3),
            "bbox": bbox,
            "verdict": verdict,
            "model": "PatchCore",
            "is_real": True,
        }
    except Exception as e:
        print(f"[WARN] Inference error: {e}, falling back to simulation")
        return simulate_predict(image_pil)


def simulate_predict(image_pil):
    """Fallback simulation when model is not available"""
    img_array = np.array(image_pil)
    mean_val = float(np.mean(img_array))
    std_val = float(np.std(img_array))

    if 100 < mean_val < 220 and std_val < 50:
        return {"type": "no defect", "confidence": 0.95 + np.random.uniform(0, 0.05),
                "anomaly_score": round(np.random.uniform(0.01, 0.15), 3),
                "verdict": "ok", "model": "simulation", "is_real": False}
    elif std_val > 60:
        score = np.random.uniform(0.4, 0.9)
        return {"type": "anomaly detected", "confidence": round(1 - score, 3),
                "anomaly_score": round(score, 3),
                "verdict": "ng" if score > THRESHOLD_WARN else "warn",
                "model": "simulation", "is_real": False}
    else:
        score = np.random.uniform(0.2, 0.5)
        return {"type": "possible anomaly", "confidence": round(1 - score, 3),
                "anomaly_score": round(score, 3),
                "verdict": "warn", "model": "simulation", "is_real": False}


# ==================== API ROUTES ====================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "AI Inspection API",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    """Core detection endpoint"""
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image file (field: image)"}), 400

    file = request.files["image"]
    start = time.time()

    try:
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_cv is None:
            return jsonify({"ok": False, "error": "Cannot decode image"}), 400

        # Convert to PIL for model input
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Run prediction
        result = predict_image(img_pil)
        elapsed = round((time.time() - start) * 1000)

        # Format response
        verdict_text = {
            "ok": "Pass",
            "warn": "Needs human confirmation",
            "ng": "Reject",
        }[result["verdict"]]

        return jsonify({
            "ok": True,
            "detections": [{
                "type": result["type"],
                "type_key": result.get("type_key", "good"),
                "type_cn": result.get("type_cn", ""),
                "confidence": result["confidence"],
                "anomaly_score": result["anomaly_score"],
                "raw_score": result.get("raw_score"),
                "is_known": result["verdict"] != "warn",
                "bbox": result.get("bbox"),
            }],
            "verdict": result["verdict"].upper(),
            "verdict_reason": verdict_text,
            "inference_time_ms": elapsed,
            "model": result["model"],
            "is_real_model": result["is_real"],
            "image_size": list(img_cv.shape[:2]),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/confirm", methods=["POST"])
def confirm():
    """Human confirmation write-back"""
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "Need JSON body"}), 400

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image_name": data.get("image_name", "unknown"),
        "ai_verdict": data.get("ai_verdict", ""),
        "ai_confidence": data.get("ai_confidence", 0),
        "defect_type": data.get("defect_type", "unknown"),
        "corrected_type": data.get("corrected_type", ""),
        "human_decision": data.get("human_decision", ""),
        "human_note": data.get("human_note", ""),
    }

    log_path = Path("./logs/confirm.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    log_count = sum(1 for _ in open(log_path)) if log_path.exists() else 0

    return jsonify({
        "ok": True,
        "message": "Confirmation recorded",
        "total_confirmations": log_count,
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Confirmation log stats"""
    log_path = Path("./logs/confirm.jsonl")
    if not log_path.exists():
        return jsonify({"ok": True, "total": 0, "by_decision": {}})

    entries = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    by_decision = {}
    for e in entries:
        d = e.get("human_decision", "unknown")
        by_decision[d] = by_decision.get(d, 0) + 1

    return jsonify({
        "ok": True,
        "total": len(entries),
        "by_decision": by_decision,
        "latest": entries[-1] if entries else None,
    })


# ==================== FRONTEND ====================

@app.route("/")
def index():
    """Serve the demo frontend (camera works because localhost is a secure context)"""
    return send_from_directory(".", "index.html")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 50)
    print("  AI Inspection · Flask API")
    print("=" * 50)
    print()

    # Load model
    model_loaded = load_model()

    print()
    print(f"  POST /api/detect    Detection")
    print(f"  POST /api/confirm   Human confirm")
    print(f"  GET  /api/stats     Log stats")
    print(f"  GET  /api/health    Health check")
    print()
    print(f"  Mode: {'REAL MODEL' if model_loaded else 'SIMULATION'}")
    print("=" * 50)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
