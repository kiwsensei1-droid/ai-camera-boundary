"""
Person Boundary Detection — Line Crossing & Tripwire Mode
- วาด "เส้นตรง 1 เส้น" (2 จุด) บนจอ เช่น เส้นหน้าประตู, เส้นกั้นห้อง, เส้นหน้าโต๊ะ
- เลือกว่าฝั่งไหนคือ "ฝั่งปลอดภัย (ในเขต)" และ "ฝั่งข้ามเส้น (นอกเขต)"
- AI (YOLOv8s + ByteTrack) จะตรวจว่าจุดเท้าของคนอยู่ฝั่งไหนของเส้น:
    * ยังไม่ข้ามเส้น = สีเขียว (ในเส้น)
    * ข้ามเกินเส้นไป = สีแดง (ออกเส้น / เกินเส้น)
- แสดง Log สีเขียว/แดง และตัวนับสถานะ Real-time รองรับทั้งคอมและมือถือ
"""

import os, io, threading, datetime, time, contextlib, base64
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

app = Flask(__name__)
app.config["SECRET_KEY"] = "pbd-line-mode-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ==================== CONFIG ====================
DEFAULT_RTSP = "rtsp://rtsp:hAtoWTdc@192.168.1.113:554/av_stream/ch0"
YOLO_MODEL_NAME = "yolov8s.pt" if os.path.exists("yolov8s.pt") else "yolov8n.pt"
YOLO_CONF    = 0.50
JPEG_QUALITY = 65

print(f"[AI Model] กำลังโหลด {YOLO_MODEL_NAME} ...")
model = YOLO(YOLO_MODEL_NAME)
with contextlib.redirect_stdout(io.StringIO()):
    model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
print(f"[AI Model] {YOLO_MODEL_NAME} พร้อมทำงาน!")

# ==================== THAI FONT ====================
_FONT_PATHS = [
    "C:/Windows/Fonts/leelawui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/THSarabunNew.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
_fcache = {}

def get_font(size: int):
    if size not in _fcache:
        for p in _FONT_PATHS:
            try:
                _fcache[size] = ImageFont.truetype(p, size)
                break
            except Exception:
                pass
        if size not in _fcache:
            _fcache[size] = ImageFont.load_default()
    return _fcache[size]

def draw_thai_labels(frame, labels):
    if not labels:
        return frame
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for text, x, y, fg, bg, sz in labels:
        fnt = get_font(sz)
        rgb_fg = (fg[2], fg[1], fg[0])
        if bg is not None:
            rgb_bg = (bg[2], bg[1], bg[0])
            bbox = draw.textbbox((x, y), text, font=fnt)
            draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill=rgb_bg)
        draw.text((x, y), text, font=fnt, fill=rgb_fg)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==================== RTSP STREAM WORKER ====================
class RTSPStreamWorker:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest_frame = None
        self.running = self.cap.isOpened()
        self.lock = threading.Lock()
        if self.running:
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.04)
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

# ==================== STATE ====================
_state_lock = threading.Lock()
_state = {
    "running": False,
    "camera_src": DEFAULT_RTSP,
    "line": [],               # [[x1, y1], [x2, y2]]
    "invert_side": False,     # สลับฝั่ง ปลอดภัย vs เกินเส้น
    "stats": {"persons": 0, "inside": 0, "outside": 0},
    "logs": []
}

_stream_worker = None
_detect_thread = None

def point_line_side(px, py, line):
    """
    ใช้ Cross Product ตรวจสอบว่าจุด (px, py) อยู่ฝั่งไหนของเส้นตรง
    คืนค่า: > 0 (ฝั่งซ้าย/บน), < 0 (ฝั่งขวา/ล่าง), 0 (บนเส้น)
    """
    (x1, y1), (x2, y2) = line[0], line[1]
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def is_person_beyond_line(foot_x, foot_y, line, invert_side=False):
    """
    line: [(x1,y1), (x2,y2)]
    ถ้าไม่มีเส้น ถือว่ายังไม่ออกเส้น (ปลอดภัย = False)
    """
    if len(line) < 2:
        return False  # ยังไม่ได้วาดเส้น
    
    val = point_line_side(foot_x, foot_y, line)
    # ค่าเริ่มต้น: val > 0 ถือว่าข้ามเส้น (เกินเส้น)
    is_crossed = (val > 0)
    if invert_side:
        is_crossed = not is_crossed
    return is_crossed

# ==================== DETECTION LOOP ====================
def detection_loop():
    global _stream_worker

    with _state_lock:
        src = _state["camera_src"]

    print(f"[Tripwire AI] กำลังเปิดกล้อง: {src}")
    _stream_worker = RTSPStreamWorker(src)
    if not _stream_worker.running:
        socketio.emit("error_msg", {"message": f"เชื่อมกล้องไม่สำเร็จ: {src}"})
        with _state_lock:
            _state["running"] = False
        return

    print("[Tripwire AI] กล้องเริ่มทำงานแล้ว!")
    socketio.emit("camera_started", {})

    prev_status_map = {}  # tid -> "IN" / "OUT"
    frame_count = 0

    while True:
        with _state_lock:
            if not _state["running"]:
                break
            line = list(_state["line"])
            invert_side = _state["invert_side"]
            conf_thresh = YOLO_CONF

        frame = _stream_worker.read()
        if frame is None:
            time.sleep(0.02)
            continue

        frame_count += 1
        fh, fw = frame.shape[:2]
        display = frame.copy()

        # รัน YOLO ตรวจเฉพาะคน
        with contextlib.redirect_stdout(io.StringIO()):
            results = model.track(
                frame,
                classes=[0],
                conf=conf_thresh,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                imgsz=640
            )

        persons = 0
        outside = 0
        current_status_map = {}
        labels = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                
                tid = int(box.id[0].item()) if box.id is not None else (i + 1)
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)

                # ตรวจว่าข้ามเส้นหรือไม่
                crossed = is_person_beyond_line(foot_x, foot_y, line, invert_side)
                status = "OUT" if crossed else "IN"
                current_status_map[tid] = status

                persons += 1
                if crossed:
                    outside += 1

                # สีตามเงื่อนไข:
                # ยังไม่ออกเส้น = สีเขียว (IN)
                # ออกเส้น/เกินเส้น = สีแดง (OUT)
                GREEN = (0, 220, 90)
                RED   = (40, 40, 255)
                box_color = RED if crossed else GREEN
                status_th = "เกินเส้น! (ออกนอกเขต)" if crossed else "อยู่ในเส้นปกติ"
                label_th = f"คน #{tid}: {status_th}"

                # วาดกรอบตัวคน + จุดตำแหน่งเท้า
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(display, (foot_x, foot_y), 7, box_color, -1)
                labels.append((label_th, x1 + 2, max(4, y1 - 24), (255, 255, 255), box_color, 16))

        # เขียนชื่อสถานะภาษาไทย
        if labels:
            display = draw_thai_labels(display, labels)

        # บันทึก Log เมื่อสถานะเปลี่ยน
        now_ts = datetime.datetime.now().strftime("%H:%M:%S")
        new_logs = []
        for tid, st in current_status_map.items():
            if st != prev_status_map.get(tid):
                if st == "OUT":
                    entry = {"ts": now_ts, "msg": f"คน #{tid} เดินข้ามเกินเส้นแล้ว! 🚨", "status": "OUT"}
                else:
                    entry = {"ts": now_ts, "msg": f"คน #{tid} ยังอยู่ในเส้นปกติ 🟢", "status": "IN"}
                new_logs.append(entry)

        prev_status_map = dict(current_status_map)

        if new_logs:
            with _state_lock:
                _state["logs"].extend(new_logs)
                _state["logs"] = _state["logs"][-300:]
            for log_item in new_logs:
                socketio.emit("log_event", log_item)

        stats_data = {"persons": persons, "inside": persons - outside, "outside": outside}
        with _state_lock:
            _state["stats"] = stats_data
        socketio.emit("stats", stats_data)

        # วาดเส้นกั้น (Line)
        if len(line) == 2:
            pt1 = (int(line[0][0]), int(line[0][1]))
            pt2 = (int(line[1][0]), int(line[1][1]))
            
            # เส้นไฮไลท์หนาชัดเจน
            line_color = (0, 0, 255) if outside > 0 else (0, 230, 255) # แดงเมื่อมีคนข้าม / เหลืองนีออนปกติ
            cv2.line(display, pt1, pt2, (0, 0, 0), 6)  # ขอบเงาดำ
            cv2.line(display, pt1, pt2, line_color, 3) # เส้นจริง
            cv2.circle(display, pt1, 8, (255, 255, 255), -1)
            cv2.circle(display, pt1, 6, (0, 140, 255), -1)
            cv2.circle(display, pt2, 8, (255, 255, 255), -1)
            cv2.circle(display, pt2, 6, (0, 140, 255), -1)

            # ตัวอักษรบอกเส้น
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            cv2.putText(display, "[ BOUNDARY LINE ]", (mid_x - 70, mid_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, "[ BOUNDARY LINE ]", (mid_x - 70, mid_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)

        # วันที่เวลา
        cv2.putText(display, datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                    (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # ส่งภาพ Base64 Stream
        if frame_count % 2 == 0:
            ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                socketio.emit("frame", b64)

    if _stream_worker:
        _stream_worker.stop()
        _stream_worker = None

    with _state_lock:
        _state["running"] = False
    socketio.emit("camera_stopped", {})
    print("[Tripwire AI] กล้องหยุดทำงานแล้ว")

# ==================== ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html", default_rtsp=DEFAULT_RTSP, model_name=YOLO_MODEL_NAME)

@app.route("/api/status")
def api_status():
    with _state_lock:
        return jsonify({
            "running": _state["running"],
            "line": _state["line"],
            "stats": _state["stats"],
            "recent_logs": _state["logs"][-30:]
        })

# ==================== SOCKET.IO ====================
@socketio.on("connect")
def on_connect():
    with _state_lock:
        emit("init_state", {
            "running": _state["running"],
            "line": _state["line"],
            "invert_side": _state["invert_side"],
            "stats": _state["stats"],
            "logs": _state["logs"][-30:]
        })

@socketio.on("start_camera")
def on_start(data):
    global _detect_thread
    with _state_lock:
        if _state["running"]:
            return
        _state["camera_src"] = data.get("source", DEFAULT_RTSP).strip() or DEFAULT_RTSP
        _state["running"] = True
        _state["logs"] = []

    _detect_thread = threading.Thread(target=detection_loop, daemon=True)
    _detect_thread.start()

@socketio.on("stop_camera")
def on_stop():
    with _state_lock:
        _state["running"] = False

@socketio.on("set_line")
def on_set_line(data):
    with _state_lock:
        _state["line"] = data.get("points", [])
    emit("line_set", {"points": _state["line"]})

@socketio.on("invert_line_side")
def on_invert_line_side():
    with _state_lock:
        _state["invert_side"] = not _state["invert_side"]
        side_status = _state["invert_side"]
    emit("line_side_inverted", {"invert": side_status})

@socketio.on("reset_line")
def on_reset_line():
    with _state_lock:
        _state["line"] = []
    emit("line_reset", {})

@socketio.on("clear_logs")
def on_clear_logs():
    with _state_lock:
        _state["logs"] = []
    emit("logs_cleared", {})

@socketio.on("set_conf")
def on_set_conf(data):
    global YOLO_CONF
    val = float(data.get("conf", 0.50))
    YOLO_CONF = max(0.15, min(0.95, val))
    emit("conf_updated", {"conf": YOLO_CONF})

# ==================== MAIN ====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  🎯 Person Boundary Line Detection (Tripwire Mode)")
    print(f"  🧠 AI Model: {YOLO_MODEL_NAME}")
    print(f"  🌐 URL เข้าดู: http://localhost:{port}")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
