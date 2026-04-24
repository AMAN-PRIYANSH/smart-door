from flask import Flask, Response, request
import subprocess
import cv2
import numpy as np
import os

app = Flask(__name__)

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

latest_frame = None


def generate():
    global latest_frame

    process = subprocess.Popen(
        ["rpicam-vid",
         "-t", "0",
         "--width", "640",
         "--height", "480",
         "--codec", "mjpeg",
         "--inline",
         "-o", "-"],
        stdout=subprocess.PIPE
    )

    buffer = b""

    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            break

        buffer += chunk

        start = buffer.find(b'\xff\xd8')
        end = buffer.find(b'\xff\xd9')

        if start != -1 and end != -1:
            jpg = buffer[start:end+2]
            buffer = buffer[end+2:]

            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # 🔥 CCTV style dynamic boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            latest_frame = frame.copy()

            _, jpeg = cv2.imencode('.jpg', frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/')
def index():
    return '''
    <html>
    <body style="text-align:center">

        <h2>Smart Door Enrollment</h2>

        <img src="/video" width="640"><br><br>

        <input id="name" placeholder="Enter name"><br><br>

        <button onclick="capture()">Capture</button>

        <div id="preview"></div>

        <script>
        function capture() {
            let name = document.getElementById("name").value;

            fetch("/capture?name=" + name)
            .then(res => res.blob())
            .then(blob => {
                let url = URL.createObjectURL(blob);
                document.getElementById("preview").innerHTML =
                    "<h3>Preview</h3><img src='" + url + "' width=300><br>" +
                    "<button onclick='save()'>Confirm</button>" +
                    "<button onclick='location.reload()'>Retake</button>";
            });
        }

        function save() {
            let name = document.getElementById("name").value;
            fetch("/save?name=" + name)
            .then(res => res.text())
            .then(alert);
        }
        </script>

    </body>
    </html>
    '''


@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


temp_frame = None


@app.route('/capture')
def capture():
    global latest_frame, temp_frame

    if latest_frame is None:
        return "No frame", 400

    temp_frame = latest_frame.copy()

    _, jpeg = cv2.imencode('.jpg', temp_frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/save')
def save():
    global temp_frame

    name = request.args.get("name")

    if temp_frame is None:
        return "No captured image"

    path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(path, temp_frame)

    return f"Saved: {path}"


app.run(host='0.0.0.0', port=5000)
