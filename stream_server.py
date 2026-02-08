import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2


class CameraStream:
    def __init__(self, camera_id: int, width: int, height: int, fps: int):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            for alt_id in [0, 1, 2]:
                if alt_id != self.camera_id:
                    self._cap = cv2.VideoCapture(alt_id, cv2.CAP_V4L2)
                    if self._cap.isOpened():
                        self.camera_id = alt_id
                        break
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ret, frame = self._cap.read()
        if not ret:
            return False
        with self._lock:
            self._frame = frame
        return True

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _loop(self) -> None:
        interval = 1.0 / self.fps
        while self._running:
            start = time.time()
            if self._cap is None:
                time.sleep(0.1)
                continue
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/mjpeg":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        try:
            while True:
                frame = self.server.camera_stream.read()
                if frame is None:
                    time.sleep(0.05)
                    continue
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                data = encoded.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / self.server.stream_fps)
        except Exception:
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    stream = CameraStream(args.camera, args.width, args.height, args.fps)
    if not stream.open():
        raise SystemExit("Failed to open camera")
    stream.start()

    server = ThreadedHTTPServer((args.host, args.port), MJPEGHandler)
    server.camera_stream = stream
    server.stream_fps = args.fps

    try:
        server.serve_forever()
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
