import subprocess
import threading
import logging
import signal
import sys
import os
import cv2
import numpy as np
import face_recognition
import smbus2
from gpiozero import MotionSensor, LED
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("smart-door")

# ─── Config ──────────────────────────────────────────────────
PIR_PIN        = 4
GREEN_PIN      = 17
RED_PIN        = 27
MLX_ADDR       = 0x5A
MIN_TEMP       = 31.0
MAX_TEMP       = 38.5
LED_DURATION   = 3.0
CAMERA_TIMEOUT = 30.0      # seconds camera stays on after last motion
FRAME_SKIP     = 5
INFER_W        = 320
INFER_H        = 240
FACE_CONF      = 0.6
MATCH_TOL      = 0.5
KNOWN_DIR      = "known_faces"
FLASK_PORT     = 5000

# ─── GPIO ────────────────────────────────────────────────────
pir       = MotionSensor(PIR_PIN)
green_led = LED(GREEN_PIN)
red_led   = LED(RED_PIN)

# ─── State ───────────────────────────────────────────────────
latest_frame   = None
frame_lock     = threading.Lock()
camera_active  = False
frame_count    = 0
last_boxes     = []
last_names     = []
led_timer      = None
camera_timer   = None
system_active  = True
camera_thread  = None
camera_process = None

# UI state
mlx_enabled    = True        # toggled from browser
last_temp      = None
last_status    = "waiting"   # waiting / active / known / unknown

# ─── LED ─────────────────────────────────────────────────────
def leds_off():
    green_led.off()
    red_led.off()

def trigger_led(which):
    global led_timer
    if led_timer:
        led_timer.cancel()
    leds_off()
    which.on()
    led_timer = threading.Timer(LED_DURATION, leds_off)
    led_timer.daemon = True
    led_timer.start()

# ─── Temperature ─────────────────────────────────────────────
def read_temp():
    global last_temp
    try:
        bus  = smbus2.SMBus(1)
        data = bus.read_word_data(MLX_ADDR, 0x07)
        temp = round((data * 0.02) - 273.15, 1)
        bus.close()
        last_temp = temp
        log.info("Temp: %.1f C", temp)
        return temp
    except Exception as e:
        log.warning("Temp read failed: %s", e)
        return None

def is_human_temp():
    temp = read_temp()
    if temp is None:
        return True        # sensor fail → don't block
    return MIN_TEMP <= temp <= MAX_TEMP

# ─── Load known faces ─────────────────────────────────────────
def load_known_faces():
    encodings, names = [], []
    if not os.path.exists(KNOWN_DIR):
        os.makedirs(KNOWN_DIR)
        return encodings, names
    for f in os.listdir(KNOWN_DIR):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(KNOWN_DIR, f)
        img  = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if not encs:
            log.warning("No face in %s — skipping", f)
            continue
        encodings.append(encs[0])
        names.append(os.path.splitext(f)[0])
        log.info("Loaded face: %s", os.path.splitext(f)[0])
    log.info("Total known faces: %d", len(names))
    return encodings, names

log.info("Loading known faces...")
known_encodings, known_names = load_known_faces()

# ─── Load YOLO ───────────────────────────────────────────────
log.info("Loading YOLOv8n-face model...")
model = YOLO("yolov8n-face.pt")
log.info("Model ready")

# ─── Flask ───────────────────────────────────────────────────
app = Flask(__name__)

# ─── Decision logic ──────────────────────────────────────────
def make_decision(names):
    global last_status
    if not names:
        return

    known = [n for n in names if n != "unknown"]

    if not known:
        log.info("Unknown face → RED")
        last_status = "unknown"
        trigger_led(red_led)
        return

    # known face found — now check MLX if enabled
    if mlx_enabled:
        if is_human_temp():
            log.info("Known + human temp → GREEN")
            last_status = "known"
            trigger_led(green_led)
        else:
            log.info("Known face but temp check failed → RED")
            last_status = "unknown"
            trigger_led(red_led)
    else:
        log.info("Known face (MLX disabled) → GREEN")
        last_status = "known"
        trigger_led(green_led)

# ─── Camera stop ─────────────────────────────────────────────
def stop_camera():
    global camera_active, camera_process, last_status
    log.info("Camera timeout — stopping")
    camera_active = False
    last_status   = "waiting"
    if camera_process:
        camera_process.terminate()
        camera_process = None
    with frame_lock:
        globals()['latest_frame'] = None

# ─── Camera timer reset ──────────────────────────────────────
def reset_camera_timer():
    global camera_timer
    if camera_timer:
        camera_timer.cancel()
    camera_timer = threading.Timer(CAMERA_TIMEOUT, stop_camera)
    camera_timer.daemon = True
    camera_timer.start()
    log.info("Camera timer reset — %.0fs remaining", CAMERA_TIMEOUT)

# ─── Camera + ML loop ────────────────────────────────────────
def camera_loop():
    global latest_frame, frame_count, last_boxes, last_names
    global camera_active, camera_process

    log.info("Camera starting")
    camera_process = subprocess.Popen(
        ["rpicam-vid",
         "-t", "0",
         "--width", "640",
         "--height", "480",
         "--framerate", "30",
         "--codec", "mjpeg",
         "--inline",
         "-o", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    buffer = b""

    while system_active and camera_active:
        chunk = camera_process.stdout.read(4096)
        if not chunk:
            break
        buffer += chunk

        start = buffer.find(b'\xff\xd8')
        end   = buffer.find(b'\xff\xd9')
        if start == -1 or end == -1:
            continue

        jpg    = buffer[start:end+2]
        buffer = buffer[end+2:]

        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        frame_count += 1

        # ── ML every FRAME_SKIP frames ──
        if frame_count % FRAME_SKIP == 0:
            small   = cv2.resize(frame, (INFER_W, INFER_H))
            results = model(small, verbose=False)[0]

            boxes   = []
            h_ratio = frame.shape[0] / INFER_H
            w_ratio = frame.shape[1] / INFER_W

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < FACE_CONF:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((
                    int(x1 * w_ratio), int(y1 * h_ratio),
                    int(x2 * w_ratio), int(y2 * h_ratio),
                    conf
                ))
            last_boxes = boxes

            names = []
            if boxes and known_encodings:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for (x1, y1, x2, y2, _) in boxes:
                    loc  = [(y1, x2, y2, x1)]
                    encs = face_recognition.face_encodings(rgb, loc)
                    if not encs:
                        names.append("unknown")
                        continue
                    dists = face_recognition.face_distance(known_encodings, encs[0])
                    best  = int(np.argmin(dists))
                    names.append(
                        known_names[best] if dists[best] <= MATCH_TOL else "unknown"
                    )
            else:
                names = ["unknown"] * len(boxes)

            last_names = names

            if names:
                threading.Thread(
                    target=make_decision, args=(names,), daemon=True
                ).start()

        # ── Draw boxes ──
        for i, (x1, y1, x2, y2, conf) in enumerate(last_boxes):
            name  = last_names[i] if i < len(last_names) else "unknown"
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}",
                        (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── LIVE tag ──
        cv2.putText(frame, "LIVE", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── Temp overlay if MLX on ──
        if mlx_enabled and last_temp is not None:
            cv2.putText(frame, f"Temp: {last_temp}C",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        _, jpeg = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            globals()['latest_frame'] = jpeg.tobytes()

    log.info("Camera loop ended")

# ─── PIR handler ─────────────────────────────────────────────
def on_motion():
    global camera_thread, camera_active, last_status
    log.info("PIR triggered")
    reset_camera_timer()
    if not camera_active:
        camera_active = True
        last_status   = "active"
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()

pir.when_motion = on_motion
log.info("PIR ready on GPIO %d", PIR_PIN)

# ─── Flask routes ─────────────────────────────────────────────
def generate_stream():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for motion...", (100, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(blank, "System armed — PIR active", (145, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            _, jpeg = cv2.imencode(".jpg", blank)
            frame = jpeg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <title>Smart Door</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      background:#0a0a0a; min-height:100vh;
      display:flex; flex-direction:column;
      align-items:center; justify-content:center;
      font-family:monospace; color:#eee; gap:16px;
      padding:16px;
    }
    h2 {
      font-size:13px; letter-spacing:4px;
      color:#555; text-transform:uppercase;
    }
    .feed {
      border:1px solid #1a1a1a; border-radius:8px;
      overflow:hidden; width:100%; max-width:640px;
    }
    img { display:block; width:100%; }

    /* status bar */
    .statusbar {
      display:flex; align-items:center; justify-content:space-between;
      width:100%; max-width:640px; gap:12px; flex-wrap:wrap;
    }
    .status-pill {
      padding:6px 14px; border-radius:20px; font-size:11px;
      letter-spacing:2px; text-transform:uppercase; font-weight:600;
      transition: all .3s;
    }
    .s-waiting { background:#111; color:#444; border:1px solid #222; }
    .s-active  { background:#0d2b0d; color:#4caf50; border:1px solid #2a5c2a; }
    .s-known   { background:#0a2e0a; color:#66ff66; border:1px solid #33aa33; }
    .s-unknown { background:#2e0a0a; color:#ff4444; border:1px solid #aa2222; }

    /* MLX toggle */
    .mlx-wrap {
      display:flex; align-items:center; gap:10px; font-size:12px; color:#666;
    }
    .toggle {
      position:relative; width:46px; height:24px;
      background:#222; border-radius:12px;
      border:1px solid #333; cursor:pointer;
      transition:background .3s;
    }
    .toggle.on { background:#1a4d1a; border-color:#2a6e2a; }
    .toggle::after {
      content:''; position:absolute;
      top:3px; left:3px;
      width:16px; height:16px;
      border-radius:50%; background:#555;
      transition:transform .3s, background .3s;
    }
    .toggle.on::after { transform:translateX(22px); background:#66ff66; }

    .temp-display {
      font-size:12px; color:#555; letter-spacing:1px;
    }
    .temp-display span { color:#0af; }

    .hint { font-size:10px; color:#333; letter-spacing:1px; }
  </style>
</head>
<body>
  <h2>Smart Door Security</h2>

  <div class="feed">
    <img src="/video" id="feed">
  </div>

  <div class="statusbar">
    <div class="status-pill s-waiting" id="status-pill">Waiting</div>

    <div class="temp-display" id="temp-display">
      Temp: <span>--.-°C</span>
    </div>

    <div class="mlx-wrap">
      <span>MLX Temp Check</span>
      <div class="toggle on" id="mlx-toggle" onclick="toggleMLX()"></div>
      <span id="mlx-label" style="color:#66ff66">ON</span>
    </div>
  </div>

  <p class="hint">green box = known face &nbsp;|&nbsp; red box = unknown</p>

  <script>
    let mlxOn = true;

    function toggleMLX() {
      mlxOn = !mlxOn;
      document.getElementById('mlx-toggle').classList.toggle('on', mlxOn);
      document.getElementById('mlx-label').textContent = mlxOn ? 'ON' : 'OFF';
      document.getElementById('mlx-label').style.color = mlxOn ? '#66ff66' : '#555';
      fetch('/mlx?enabled=' + mlxOn);
    }

    // poll status every 1.5s
    function pollStatus() {
      fetch('/status')
        .then(r => r.json())
        .then(d => {
          const pill = document.getElementById('status-pill');
          const map  = {
            waiting: ['Waiting',  's-waiting'],
            active:  ['Camera ON','s-active'],
            known:   ['Access OK','s-known'],
            unknown: ['UNKNOWN',  's-unknown'],
          };
          const [label, cls] = map[d.status] || map.waiting;
          pill.textContent = label;
          pill.className   = 'status-pill ' + cls;

          if (d.temp !== null) {
            document.getElementById('temp-display').innerHTML =
              'Temp: <span>' + d.temp + '°C</span>';
          }
        })
        .catch(() => {});
    }
    setInterval(pollStatus, 1500);
    pollStatus();
  </script>
</body>
</html>
    """)

@app.route("/video")
def video():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/mlx")
def mlx_toggle():
    global mlx_enabled
    val        = request.args.get("enabled", "true").lower()
    mlx_enabled = val == "true"
    log.info("MLX check %s", "ON" if mlx_enabled else "OFF")
    return jsonify({"mlx": mlx_enabled})

@app.route("/status")
def status():
    return jsonify({
        "status": last_status,
        "temp":   last_temp,
        "mlx":    mlx_enabled,
    })

# ─── Graceful shutdown ────────────────────────────────────────
def shutdown(sig=None, frame=None):
    global system_active
    log.info("Shutting down...")
    system_active = False
    if led_timer:    led_timer.cancel()
    if camera_timer: camera_timer.cancel()
    leds_off()
    pir.close()
    green_led.close()
    red_led.close()
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ─── Start ───────────────────────────────────────────────────
if __name__ == "__main__":
    from flask import request
    log.info("Smart door ready")
    log.info("Open http://<pi-ip>:%d in browser", FLASK_PORT)
    log.info("Waiting for motion...")
    app.run(host="0.0.0.0", port=FLASK_PORT,
            debug=False, threaded=True, use_reloader=False)
