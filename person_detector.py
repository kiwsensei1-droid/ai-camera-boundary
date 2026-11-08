"""
Person Detection with Zone Boundary Monitoring
==============================================
ตรวจจับคนในพื้นที่กล้อง พร้อมแจ้งเตือนเมื่อออกนอกเขต

Controls:
  - คลิก 4 จุด เพื่อวาด Zone polygon (ครั้งแรกที่รัน)
  - r : รีเซ็ต Zone ใหม่
  - q : ออกจากโปรแกรม
  - s : บันทึก screenshot
  - SPACE : หยุด/เล่น
"""

import cv2
import numpy as np
import datetime
import os
import sys

# ==================== CONFIG ====================
CAMERA_SOURCE = 0          # 0 = webcam, หรือใส่ path ไฟล์วิดีโอ เช่น "video.mp4"
ZONE_COLOR_IN  = (0, 255, 0)      # สีเขียว = ในเขต
ZONE_COLOR_OUT = (0, 0, 255)      # สีแดง   = นอกเขต
ZONE_ALPHA     = 0.3              # ความโปร่งแสงของ Zone
LOG_MAX_LINES  = 20               # จำนวน log บรรทัดสูงสุดบนหน้าจอ
HOG_WIN_STRIDE = (8, 8)
HOG_PADDING    = (4, 4)
HOG_SCALE      = 1.05
MIN_CONFIDENCE = 0.3              # กรองผล HOG (ยิ่งสูง ยิ่งเข้มงวด)
LOG_FILE       = "detection_log.txt"
# ================================================

# ANSI colors for terminal
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"


def log_print(message: str, color: str = RESET):
    """พิมพ์ log บน terminal พร้อมสี"""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    print(f"{color}{line}{RESET}")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def point_in_polygon(point, polygon):
    """ตรวจสอบว่า point อยู่ใน polygon หรือไม่"""
    if len(polygon) < 3:
        return True  # ถ้ายังไม่ได้ตั้ง zone ถือว่าอยู่ใน
    result = cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False)
    return result >= 0


def get_foot_point(box):
    """จุดเท้า (ตรงกลางด้านล่าง) ของ bounding box"""
    x, y, w, h = box
    return (int(x + w / 2), int(y + h))


def draw_zone(frame, zone_points, any_outside):
    """วาด Zone polygon บน frame"""
    if len(zone_points) < 3:
        return frame
    overlay = frame.copy()
    pts = np.array(zone_points, dtype=np.int32)
    color = ZONE_COLOR_OUT if any_outside else ZONE_COLOR_IN
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)
    return frame


def draw_log_panel(frame, logs):
    """วาด log panel ด้านขวาของหน้าจอ"""
    h, w = frame.shape[:2]
    panel_w = 380
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    cv2.putText(panel, "[ DETECTION LOG ]", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.line(panel, (5, 32), (panel_w - 5, 32), (80, 80, 80), 1)

    for i, (ts, msg, status) in enumerate(logs[-LOG_MAX_LINES:]):
        y = 55 + i * 22
        if y > h - 10:
            break
        color = (0, 220, 0) if status == "IN" else (60, 60, 255)
        icon  = "✔" if status == "IN" else "✖"
        text  = f"{ts}  {icon} {msg}"
        cv2.putText(panel, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    combined = np.hstack([frame, panel])
    return combined


def draw_hud(frame, zone_points, person_count, outside_count, paused):
    """วาด HUD ข้อมูลบน frame"""
    h, w = frame.shape[:2]
    # กล่องข้อมูลบนซ้าย
    cv2.rectangle(frame, (0, 0), (260, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (260, 100), (60, 60, 60), 1)

    zone_status = "ตั้งค่าแล้ว" if len(zone_points) >= 3 else f"คลิก {4 - len(zone_points)} จุดอีก"
    cv2.putText(frame, f"Zone: {zone_status}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"พบคน: {person_count} คน", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    inside = person_count - outside_count
    in_color  = (0, 220, 0)
    out_color = (60, 60, 255)
    cv2.putText(frame, f"ในเขต: {inside}", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, in_color, 1)
    cv2.putText(frame, f"นอกเขต: {outside_count}", (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, out_color, 1)

    # วาง keys hint ล่างซ้าย
    hints = "[Q]ออก  [R]รีเซ็ต Zone  [S]Screenshot  [SPACE]หยุด"
    cv2.rectangle(frame, (0, h - 22), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, hints, (5, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    if paused:
        cv2.putText(frame, "⏸ PAUSED", (w // 2 - 55, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
    return frame


# ============================================================
#                         MAIN
# ============================================================
def main():
    log_print("=== Person Boundary Detection เริ่มทำงาน ===", CYAN)
    log_print(f"Camera source: {CAMERA_SOURCE}", CYAN)

    # HOG Detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        log_print(f"❌ ไม่สามารถเปิด camera source: {CAMERA_SOURCE}", RED)
        sys.exit(1)

    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30
    log_print(f"Camera FPS: {fps_cap:.1f}", CYAN)

    zone_points   = []   # polygon points (x, y)
    logs          = []   # [(time_str, message, "IN"/"OUT")]
    paused        = False
    frame_count   = 0
    last_statuses = {}   # person_id -> "IN"/"OUT" (ใช้ index แทน id ในรอบนี้)

    # ---- Mouse callback สำหรับ click zone ----
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(zone_points) < 4:
            zone_points.append((x, y))
            log_print(f"Zone จุดที่ {len(zone_points)}: ({x}, {y})", YELLOW)

    cv2.namedWindow("Person Detector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Person Detector", mouse_cb)

    log_print("💡 คลิก 4 จุดบน preview เพื่อกำหนดพื้นที่ Zone", YELLOW)
    log_print("   กด [R] เพื่อรีเซ็ต | [Q] ออก | [SPACE] หยุด/เล่น", YELLOW)

    ret, frame = cap.read()
    if not ret:
        log_print("❌ อ่านเฟรมแรกไม่ได้", RED)
        sys.exit(1)

    frame_h, frame_w = frame.shape[:2]
    ss_dir = "screenshots"
    os.makedirs(ss_dir, exist_ok=True)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # ถ้าเป็นวิดีโอ loop กลับต้น
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            frame_count += 1

        display = frame.copy()
        any_outside = False
        person_count = 0
        outside_count = 0
        current_statuses = {}

        # ---- ตรวจจับคนทุก 2 เฟรม (ประหยัด CPU) ----
        if not paused and frame_count % 2 == 0:
            small = cv2.resize(frame, (frame_w // 2, frame_h // 2))
            rects, weights = hog.detectMultiScale(
                small,
                winStride=HOG_WIN_STRIDE,
                padding=HOG_PADDING,
                scale=HOG_SCALE
            )

            if len(rects) > 0:
                # กรองด้วย confidence และ scale กลับ
                rects_full = []
                for i, (rx, ry, rw, rh) in enumerate(rects):
                    if len(weights) > 0 and weights[i] < MIN_CONFIDENCE:
                        continue
                    rects_full.append((rx * 2, ry * 2, rw * 2, rh * 2))

                # Non-Maximum Suppression
                if rects_full:
                    rects_np = np.array([[x, y, x + w, y + h] for x, y, w, h in rects_full])
                    pick = []
                    x1, y1, x2, y2 = rects_np[:, 0], rects_np[:, 1], rects_np[:, 2], rects_np[:, 3]
                    area  = (x2 - x1 + 1) * (y2 - y1 + 1)
                    idxs  = np.argsort(y2)
                    while len(idxs) > 0:
                        last = len(idxs) - 1
                        i    = idxs[last]
                        pick.append(i)
                        xx1  = np.maximum(x1[i], x1[idxs[:last]])
                        yy1  = np.maximum(y1[i], y1[idxs[:last]])
                        xx2  = np.minimum(x2[i], x2[idxs[:last]])
                        yy2  = np.minimum(y2[i], y2[idxs[:last]])
                        w_   = np.maximum(0, xx2 - xx1 + 1)
                        h_   = np.maximum(0, yy2 - yy1 + 1)
                        overlap = (w_ * h_) / area[idxs[:last]]
                        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > 0.65)[0])))

                    for pid, i in enumerate(pick):
                        rx, ry, rw, rh = rects_full[i]
                        foot = get_foot_point((rx, ry, rw, rh))
                        inside = point_in_polygon(foot, zone_points) if len(zone_points) >= 3 else True

                        status    = "IN" if inside else "OUT"
                        box_color = ZONE_COLOR_IN if inside else ZONE_COLOR_OUT
                        label     = f"Person {pid+1}: {'ในเขต' if inside else 'นอกเขต!'}"

                        # วาด bounding box
                        cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), box_color, 2)
                        cv2.circle(display, foot, 5, box_color, -1)
                        # label background
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(display, (rx, ry - th - 8), (rx + tw + 4, ry), box_color, -1)
                        cv2.putText(display, label, (rx + 2, ry - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        person_count += 1
                        current_statuses[pid] = status
                        if not inside:
                            outside_count += 1
                            any_outside = True

                # --- Log เฉพาะเมื่อ status เปลี่ยน ---
                for pid, status in current_statuses.items():
                    prev = last_statuses.get(pid)
                    ts_str = datetime.datetime.now().strftime("%H:%M:%S")
                    if status != prev:
                        if status == "OUT":
                            msg = f"Person {pid+1} ออกนอกเขต!"
                            log_print(msg, RED)
                            logs.append((ts_str, msg, "OUT"))
                        else:
                            msg = f"Person {pid+1} อยู่ในเขต"
                            log_print(msg, GREEN)
                            logs.append((ts_str, msg, "IN"))
                last_statuses = current_statuses.copy()

        # ---- วาด Zone ----
        draw_zone(display, zone_points, any_outside)

        # วาดจุดที่กำลัง click zone
        for i, pt in enumerate(zone_points):
            cv2.circle(display, pt, 6, (0, 255, 255), -1)
            if i > 0:
                cv2.line(display, zone_points[i - 1], pt, (0, 255, 255), 2)
        if len(zone_points) == 4:
            cv2.line(display, zone_points[-1], zone_points[0], (0, 255, 255), 2)

        # ---- HUD ----
        draw_hud(display, zone_points, person_count, outside_count, paused)

        # ---- Log Panel ----
        combined = draw_log_panel(display, logs)

        cv2.imshow("Person Detector", combined)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log_print("ออกจากโปรแกรม", CYAN)
            break
        elif key == ord('r'):
            zone_points.clear()
            last_statuses.clear()
            log_print("รีเซ็ต Zone แล้ว — คลิก 4 จุดใหม่", YELLOW)
        elif key == ord('s'):
            fname = os.path.join(ss_dir, f"ss_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(fname, combined)
            log_print(f"บันทึก screenshot: {fname}", CYAN)
        elif key == ord(' '):
            paused = not paused
            log_print("⏸ PAUSED" if paused else "▶ RESUME", YELLOW)

    cap.release()
    cv2.destroyAllWindows()
    log_print(f"Log บันทึกที่: {os.path.abspath(LOG_FILE)}", CYAN)


if __name__ == "__main__":
    main()
