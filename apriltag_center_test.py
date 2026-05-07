#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apriltag_center_test.py

Raspberry Pi 카메라로 AprilTag를 감지하고
태그가 화면 중앙에 위치했는지 판단하는 독립 테스트 스크립트.

드론/FC/SITL 불필요 — RPi + 카메라만으로 실행 가능.

설치:
    pip install opencv-python pupil-apriltags

실행:
    python apriltag_center_test.py

    # 카메라 인덱스가 다를 경우
    CAMERA_INDEX=1 python apriltag_center_test.py

    # 화면 출력 없이 터미널 로그만 볼 경우 (헤드리스)
    HEADLESS=1 python apriltag_center_test.py
"""

import cv2
import math
import os
import time
from datetime import datetime
from pupil_apriltags import Detector

# ── 설정값 (필요 시 수정) ───────────────────────────────
CAMERA_INDEX   = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_W        = 640
FRAME_H        = 480
FPS_TARGET     = 30
TAG_FAMILY     = "tag36h11"

# 중앙 판정 임계값 (픽셀)
# → 태그 중심이 화면 중심으로부터 이 값 이내면 "중앙" 판정
CENTER_TH_PX   = 20

# 중앙 판정 유지 시간 (초)
# → 이 시간 동안 계속 중앙이어야 "안정" 판정
CENTER_HOLD_SEC = 1.0

# 헤드리스 모드 (SSH 환경 등 화면 없을 때)
HEADLESS       = os.getenv("HEADLESS", "0") == "1"

# ── 색상 정의 ───────────────────────────────────────────
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (180, 180, 180)


def open_camera() -> cv2.VideoCapture:
    # RPi 카메라: 먼저 libcamera 백엔드 시도
    for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 최신 프레임만 유지
            print(f"[camera] 연결 성공: index={CAMERA_INDEX}, backend={backend}")
            return cap
    raise RuntimeError(f"카메라를 열 수 없습니다. CAMERA_INDEX={CAMERA_INDEX}")


def detect_tag(detector: Detector, frame):
    """
    태그 감지 후 화면 중심 기준 오프셋 반환.
    반환: (cx, cy, corners, tag_id) 또는 None
      cx, cy: 화면 중심 기준 픽셀 오프셋 (양수=우/하, 음수=좌/상)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    if not tags:
        return None

    # 여러 태그 중 화면에서 가장 큰 태그 선택
    tag = max(tags, key=lambda t: cv2.contourArea(
        t.corners.astype("float32").reshape(-1, 1, 2)
    ))

    cx = tag.center[0] - FRAME_W / 2
    cy = tag.center[1] - FRAME_H / 2
    return cx, cy, tag.corners, tag.tag_id


def judge_center(cx: float, cy: float) -> tuple:
    """
    중앙 판정.
    반환: (is_centered, dist_px, direction)
      is_centered : 임계값 이내 여부
      dist_px     : 화면 중심까지 거리 (픽셀)
      direction   : 보정 방향 힌트 문자열
    """
    dist = math.hypot(cx, cy)
    is_centered = dist <= CENTER_TH_PX

    # 방향 힌트 (8방향)
    if is_centered:
        direction = "중앙 ✅"
    else:
        v = "↑ 위" if cy < -10 else ("↓ 아래" if cy > 10 else "")
        h = "← 좌" if cx < -10 else ("→ 우"  if cx > 10  else "")
        direction = f"{v} {h}".strip() or "근접"

    return is_centered, dist, direction


def draw_overlay(frame, result, is_centered, dist, direction,
                 center_held_sec, stable):
    """
    프레임에 오버레이 그리기:
    - 화면 중심 십자선
    - 중앙 판정 원
    - 태그 윤곽선 + 중심점
    - 상태 텍스트
    """
    cx_screen = FRAME_W // 2
    cy_screen = FRAME_H // 2

    # 화면 중심 십자선
    cv2.line(frame, (cx_screen - 20, cy_screen),
             (cx_screen + 20, cy_screen), COLOR_GRAY, 1)
    cv2.line(frame, (cx_screen, cy_screen - 20),
             (cx_screen, cy_screen + 20), COLOR_GRAY, 1)

    # 중앙 판정 원 (임계값 시각화)
    circle_color = COLOR_GREEN if is_centered else COLOR_RED
    cv2.circle(frame, (cx_screen, cy_screen),
               CENTER_TH_PX, circle_color, 1)

    if result:
        cx_off, cy_off, corners, tag_id = result

        # 태그 윤곽선
        pts = corners.astype(int)
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)],
                      True, circle_color, 2)

        # 태그 중심점
        tag_cx = int(cx_screen + cx_off)
        tag_cy = int(cy_screen + cy_off)
        cv2.circle(frame, (tag_cx, tag_cy), 5, circle_color, -1)

        # 태그 중심 → 화면 중심 선
        cv2.line(frame, (tag_cx, tag_cy),
                 (cx_screen, cy_screen), COLOR_YELLOW, 1)

        # 태그 ID
        cv2.putText(frame, f"ID:{tag_id}",
                    (pts[0][0], pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # 상태 패널 (좌상단)
    lines = [
        f"offset: ({cx_off if result else 0:.1f}, "
        f"{cy_off if result else 0:.1f}) px",
        f"dist  : {dist:.1f} px  [기준: {CENTER_TH_PX}px]",
        f"방향  : {direction}",
        f"유지  : {center_held_sec:.1f}s / {CENTER_HOLD_SEC}s",
        "[ 안정 ✅ ]" if stable else "",
    ]
    for i, line in enumerate(lines):
        if not line:
            continue
        color = COLOR_GREEN if (i == 4) else COLOR_WHITE
        cv2.putText(frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # 태그 미감지 표시
    if not result:
        cv2.putText(frame, "태그 미감지",
                    (cx_screen - 60, cy_screen + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

    return frame


def log_status(result, is_centered, dist, direction,
               center_held_sec, stable, fps):
    """터미널 로그 출력"""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    if result:
        cx, cy, _, tag_id = result
        status = "✅ 안정" if stable else ("🎯 중앙" if is_centered else "❌ 벗어남")
        print(
            f"[{ts}] {status}  "
            f"ID={tag_id}  "
            f"offset=({cx:+.1f}, {cy:+.1f})px  "
            f"dist={dist:.1f}px  "
            f"방향={direction}  "
            f"유지={center_held_sec:.1f}s  "
            f"FPS={fps:.1f}"
        )
    else:
        print(f"[{ts}] ❌ 태그 미감지  FPS={fps:.1f}")


def main():
    print("=" * 55)
    print("  AprilTag 중심 판정 테스트")
    print(f"  카메라 인덱스 : {CAMERA_INDEX}")
    print(f"  해상도        : {FRAME_W}x{FRAME_H}")
    print(f"  중심 임계값   : {CENTER_TH_PX}px")
    print(f"  안정 유지시간 : {CENTER_HOLD_SEC}s")
    print(f"  헤드리스 모드 : {HEADLESS}")
    print("=" * 55)
    print("종료: q 키 또는 Ctrl+C\n")

    cap      = open_camera()
    detector = Detector(families=TAG_FAMILY, nthreads=2)

    center_ok_since = None   # 중앙 진입 시각
    stable          = False  # 안정 판정 여부
    prev_time       = time.time()
    fps             = 0.0

    # 로그 출력 간격 (너무 빠르면 터미널이 지저분해짐)
    LOG_INTERVAL    = 0.2
    last_log_time   = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[camera] 프레임 읽기 실패")
                time.sleep(0.1)
                continue

            # FPS 계산
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # 태그 감지
            result = detect_tag(detector, frame)

            # 중심 판정
            if result:
                cx, cy, corners, tag_id = result
                is_centered, dist, direction = judge_center(cx, cy)
            else:
                cx = cy = 0.0
                is_centered, dist, direction = False, 0.0, "미감지"

            # 중앙 유지 시간 계산
            if is_centered:
                if center_ok_since is None:
                    center_ok_since = now
                center_held_sec = now - center_ok_since
                stable = center_held_sec >= CENTER_HOLD_SEC
            else:
                center_ok_since = None
                center_held_sec = 0.0
                stable          = False

            # 안정 판정 시 터미널 강조
            if stable:
                print(f"\n{'='*55}")
                print(f"  ✅ 태그가 중앙에 안정적으로 위치했습니다!")
                print(f"  offset=({cx:+.1f}, {cy:+.1f})px  "
                      f"dist={dist:.1f}px  유지={center_held_sec:.1f}s")
                print(f"{'='*55}\n")
                stable = False          # 한 번만 출력
                center_ok_since = now   # 리셋 후 재카운트

            # 터미널 로그 (일정 간격)
            if now - last_log_time >= LOG_INTERVAL:
                log_status(result, is_centered, dist, direction,
                           center_held_sec, stable, fps)
                last_log_time = now

            # 화면 출력 (헤드리스 아닐 때)
            if not HEADLESS:
                draw_overlay(
                    frame, result, is_centered, dist, direction,
                    center_held_sec, stable
                )
                cv2.imshow("AprilTag Center Test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[종료] q 키 입력")
                    break

    except KeyboardInterrupt:
        print("\n[종료] Ctrl+C 입력")
    finally:
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("[camera] 카메라 해제 완료")


if __name__ == "__main__":
    main()
