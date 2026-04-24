# ─── GPIO Pins ───────────────────────────────────────────────
PIR_PIN       = 4   # Pin 7
GREEN_LED_PIN = 17  # Pin 11
RED_LED_PIN   = 27  # Pin 13

# ─── I2C (MLX90614) ──────────────────────────────────────────
MLX_ADDRESS   = 0x5A  # default I2C address

# ─── Camera ──────────────────────────────────────────────────
CAMERA_WIDTH      = 640
CAMERA_HEIGHT     = 480
CAMERA_FPS        = 30
FRAME_SKIP        = 5        # run ML every Nth frame
INFER_WIDTH       = 320      # resize frame before ML inference
INFER_HEIGHT      = 240

# ─── ML / Recognition ────────────────────────────────────────
KNOWN_FACES_DIR       = "known_faces"
FACE_MATCH_TOLERANCE  = 0.5   # lower = stricter match
MIN_FACE_CONFIDENCE   = 0.6   # YOLOv8 detection threshold

# ─── Temperature ─────────────────────────────────────────────
MIN_HUMAN_TEMP = 30.0   # °C  below this = not a real human
MAX_HUMAN_TEMP = 38.5   # °C  above this = fever / fake reading

# ─── LED timing ──────────────────────────────────────────────
LED_ON_DURATION = 3.0   # seconds to keep LED on after decision

# ─── Flask ───────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# ─── System ──────────────────────────────────────────────────
LOG_LEVEL = "INFO"
