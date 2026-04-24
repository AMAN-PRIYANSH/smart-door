# 🚪 Smart Door Security System

A real-time face rec

ognition based door security system built on **Raspberry Pi 4** using computer vision, infrared temperature sensing, and a live Flask web interface.

---

## 📸 Demo

> Live face detection with bounding boxes, temperature check, and LED indicators — all accessible from a browser on the same network.


https://github.com/user-attachments/assets/c533f05d-0d3c-446f-bbbc-65c1a1efac11


---

## 🧩 Hardware Used

| Component | Purpose |
|---|---|
| Raspberry Pi 4 Model B | Central processing unit |
| Pi Camera Module (CSI) | Live video capture |
| PIR Sensor (HC-SR501) | Motion detection — activates camera |
| MLX90614 (I2C) | Infrared temperature — liveness check |
| Green LED + 220Ω resistor | Access granted indicator |
| Red LED + 220Ω resistor | Access denied indicator |
| Breadboard + Jumper wires | Circuit assembly |

---

## ⚙️ How It Works

```
PIR detects motion
      ↓
Camera activates for 30 seconds (resets on new motion)
      ↓
YOLOv8n-face detects faces every 5th frame
      ↓
face_recognition matches against known_faces/
      ↓
MLX90614 checks temperature (toggleable from UI)
      ↓
Known face + human temp → GREEN LED + "Access OK"
Unknown face or cold temp → RED LED + "UNKNOWN"
```

---

## 📁 Project Structure

```
smart-door/
├── main_web.py                        ← main system (run this)
├── enroll_web.py                      ← face enrollment tool
├── config.py                          ← all settings and GPIO pins
├── haarcascade_frontalface_default.xml← face detection for enrollment
├── yolov8n-face.pt                    ← YOLOv8 face model weights
├── requirements.txt                   ← Python dependencies
└── known_faces/                       ← enrolled face images (gitignored)
```

---

## 🔌 GPIO Wiring

| Pi Pin | Label | Connected To |
|---|---|---|
| Pin 1 | 3.3V | MLX90614 VCC |
| Pin 2 | 5V | PIR VCC |
| Pin 3 | GPIO 2 (SDA) | MLX90614 SDA |
| Pin 5 | GPIO 3 (SCL) | MLX90614 SCL |
| Pin 6 | GND | PIR GND + LED cathodes |
| Pin 7 | GPIO 4 | PIR signal OUT |
| Pin 11 | GPIO 17 | Green LED anode |
| Pin 13 | GPIO 27 | Red LED anode |
| CSI | — | Pi Camera ribbon cable |

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone git@github.com:AMAN-PRIYANSH/smart-door.git
cd smart-door
```

### 2. Enable I2C and Camera on Pi
```bash
sudo raspi-config
# Interface Options → I2C → Enable
# Interface Options → Camera → Enable
sudo reboot
```

### 3. Install dependencies
```bash
sudo apt install -y cmake libopenblas-dev liblapack-dev python3-dev libjpeg-dev i2c-tools
pip3 install -r requirements.txt --break-system-packages
```

> ⚠️ `dlib` compiles from source on Pi — takes 20–30 minutes. Leave it running.

### 4. Enroll faces
```bash
python3 enroll_web.py
# Open http://<pi-ip>:5000 in browser
# Enter name → Capture → Confirm
```

### 5. Run the system
```bash
python3 main_web.py
# Open http://<pi-ip>:5000 in browser
```

---

## 🌐 Web Interface

- **Live stream** — MJPEG feed with bounding boxes
- **Status pill** — Waiting / Camera ON / Access OK / UNKNOWN
- **Temperature display** — real-time MLX90614 reading
- **MLX toggle** — enable or disable temperature check on the fly

---

## 📦 Dependencies

```
ultralytics       ← YOLOv8 face detection
face_recognition  ← dlib-based face matching
opencv-python-headless ← frame processing
flask             ← web server and streaming
gpiozero          ← GPIO control (PIR + LEDs)
smbus2            ← I2C for MLX90614
```

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Stream FPS | ~30 fps |
| ML inference | Every 5th frame |
| Inference resolution | 320×240 (resized) |
| Best case detection | ~450ms |
| Average detection | ~900ms |
| Worst case | ~2.5s |

---

## 👥 Team

Built as part of MPCA course project at **PES University**.

---

## 📄 License

MIT
