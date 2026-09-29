# PATH: project/main.py
"""
TL;DR
- Script completo (Python 3.10/3.11) para procesar un video llamado `carretera` (por defecto `carretera.mp4`),
  detectar vehículos y recortar placas con un detector YOLOv8 (modelo de placa propio),
  extraer texto con EasyOCR y comparar contra `busquedas.json` para emitir avisos.

Ahora con soporte GPU automático (CUDA) para YOLOv8 y EasyOCR.
"""

import os
import time
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch                     # ✅ para verificar GPU
from ultralytics import YOLO
import easyocr
from difflib import SequenceMatcher

# --------------------------- Config / Defaults ---------------------------
VIDEO_DEFAULT = "CAMIONETA.mp4"
PLATE_MODEL_DEFAULT = "C:\\Users\\mateu\\OneDrive\\Documentos\\License_plate_detector&monitoreo\\yolov8-license-plate-detection\\best.pt"
VEHICLE_MODEL_DEFAULT = "yolov8n.pt"            # opcional, para detección de vehículos
BUSQUEDAS_JSON = "busquedas.json"
OUTPUT_VIDEO = "out_annotated.mp4"
COOLDOWN_SECONDS = 30
OCR_LANGS = ["es", "en"]
IMG_SIZE = 1280

# --------------------------- Util helpers ---------------------------

def ensure_busquedas_json(path: str):
    p = Path(path)
    if not p.exists():
        default = {"target_plates": ["ABC1234", "XYZ987"]}
        p.write_text(json.dumps(default, indent=2))
        print(f"Creado {path} de ejemplo. Editalo con las matriculas en busqueda.")


def load_busquedas(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    targets = [normalize_plate(x) for x in data.get("target_plates", [])]
    return set(targets)


def normalize_plate(s: str) -> str:
    if not s:
        return ""
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def is_target_plate(plate: str, targets: set, fuzzy_threshold=0.8) -> (bool, float, str):
    p = normalize_plate(plate)
    if not p:
        return False, 0.0, ""
    if p in targets:
        return True, 1.0, p
    best = (0.0, "")
    for t in targets:
        r = similar(p, t)
        if r > best[0]:
            best = (r, t)
    if best[0] >= fuzzy_threshold:
        return True, best[0], best[1]
    return False, best[0], best[1]

# --------------------------- Main Processing ---------------------------

def process_video(video_path: str,
                  plate_model_path: str = PLATE_MODEL_DEFAULT,
                  vehicle_model_path: str = VEHICLE_MODEL_DEFAULT,
                  busquedas_json: str = BUSQUEDAS_JSON,
                  output_path: str = OUTPUT_VIDEO,
                  show_window: bool = True):

    ensure_busquedas_json(busquedas_json)
    targets = load_busquedas(busquedas_json)
    print(f"Placas en busqueda cargadas: {targets}")

    # ✅ Verificar GPU
    use_gpu = torch.cuda.is_available()
    device_str = "cuda" if use_gpu else "cpu"
    print(f"Dispositivo seleccionado: {device_str}")

    # Cargar modelos
    print("Cargando modelos YOLO...")
    try:
        plate_model = YOLO(plate_model_path)
        plate_model.to(device_str)   # ✅ Forzar a GPU si está disponible
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo de placas en {plate_model_path}: {e}")

    vehicle_model = None
    try:
        vehicle_model = YOLO(vehicle_model_path)
        vehicle_model.to(device_str) # ✅ Forzar a GPU si está disponible
    except Exception:
        print("No se pudo cargar modelo de vehiculos (opcional). Continuando sin el.")

    # EasyOCR reader
    print("Inicializando EasyOCR...")
    print(f"GPU disponible para EasyOCR: {use_gpu}")
    reader = easyocr.Reader(OCR_LANGS, gpu=use_gpu)   # ✅ GPU automático

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_report_time = {}
    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # (Optional) detect vehicles
            if vehicle_model is not None:
                try:
                    vres = vehicle_model(frame, imgsz=IMG_SIZE, conf=0.25, device=device_str)
                    for r in vres:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)
                except Exception:
                    pass

            # detect plates
            pres = plate_model(frame, imgsz=IMG_SIZE, conf=0.35, device=device_str)
            for r in pres:
                for box, score in zip(r.boxes, r.boxes.conf):
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    except Exception:
                        continue
                    w = x2 - x1
                    h = y2 - y1
                    pad_x = int(w * 0.15)
                    pad_y = int(h * 0.25)
                    nx1 = max(0, x1 - pad_x)
                    ny1 = max(0, y1 - pad_y)
                    nx2 = min(frame.shape[1], x2 + pad_x)
                    ny2 = min(frame.shape[0], y2 + pad_y)
                    crop = frame[ny1:ny2, nx1:nx2]

                    if crop.size == 0:
                        continue

                    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop_resized = cv2.resize(crop_gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                    try:
                        ocr_result = reader.readtext(crop_resized)
                    except Exception:
                        ocr_result = []

                    texts = [t[1] for t in ocr_result if len(t) >= 2]
                    full_text = " ".join(texts).strip()
                    norm = normalize_plate(full_text)

                    matched, score, matched_target = is_target_plate(norm, targets, fuzzy_threshold=0.8)

                    label = norm if norm else (full_text if full_text else "--")
                    color = (0, 255, 0) if matched else (0, 180, 255)

                    cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), color, 2)
                    cv2.putText(frame, f"{label}", (nx1, ny1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    now = time.time()
                    if matched:
                        last = last_report_time.get(norm)
                        if last is None or (now - last) > COOLDOWN_SECONDS:
                            last_report_time[norm] = now
                            send_alert(norm, matched_target, score, frame_idx, (nx1, ny1, nx2, ny2))

            writer.write(frame)
            if show_window:
                cv2.imshow("ANPR", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"Procesado {frame_idx} frames en {time.time()-start_time:.1f}s. Archivo de salida: {output_path}")

# --------------------------- Alert / Reporting ---------------------------

def send_alert(detected_plate: str, matched_target: str, score: float, frame_idx: int, bbox: tuple):
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"[ALERTA] {ts} - Matricula detectada: '{detected_plate}' matched '{matched_target}' (score={score:.2f}) at frame {frame_idx} bbox={bbox}")

# --------------------------- CLI ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ANPR pipeline: detectar placas en video y comparar con busquedas.json")
    parser.add_argument('--video', type=str, default=VIDEO_DEFAULT, help='Ruta al video (ej: carretera.mp4)')
    parser.add_argument('--plate-model', type=str, default=PLATE_MODEL_DEFAULT, help='Ruta al .pt del detector de placas')
    parser.add_argument('--vehicle-model', type=str, default=VEHICLE_MODEL_DEFAULT, help='Ruta al modelo YOLO de vehiculos (opcional)')
    parser.add_argument('--busquedas', type=str, default=BUSQUEDAS_JSON, help='JSON con matriculas en busqueda')
    parser.add_argument('--output', type=str, default=OUTPUT_VIDEO, help='Archivo de salida anotado')
    parser.add_argument('--no-window', dest='show', action='store_false', help='No mostrar ventana GUI')
    args = parser.parse_args()

    process_video(args.video, args.plate_model, args.vehicle_model, args.busquedas, args.output, show_window=args.show)
