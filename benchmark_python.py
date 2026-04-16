import time
import json
import sys
import cv2
from ultralytics import YOLO

WINDOW_NAME = "Benchmark — live inference (press q to quit early)"
NUM_FRAMES  = 100
INPUT_SIZE  = 512
CLASS_NAMES = ["drill", "hammer", "pliers", "screwdriver", "wrench"]

# Colour palette per class (BGR)
COLOURS = [
    (0, 165, 255),   # drill        – orange
    (0, 255,   0),   # hammer       – green
    (255,  0,   0),  # pliers       – blue
    (0,   0, 255),   # screwdriver  – red
    (255,  0, 255),  # wrench       – magenta
]

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model best.pt ...")
model = YOLO("best.pt")

# ── Open webcam ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam (device 0). "
          "Check System Settings → Privacy & Security → Camera.")
    sys.exit(1)

# macOS AVFoundation needs a couple of frames before it starts delivering pixels.
# Drain up to 10 frames silently so the real benchmark doesn't get a black frame.
print("Warming up camera ...")
for _ in range(10):
    ret, _ = cap.read()
    if ret:
        break
    time.sleep(0.1)

# Verify at least one real frame is available before starting the timed loop.
ret, test_frame = cap.read()
if not ret or test_frame is None:
    cap.release()
    print("ERROR: Camera opened but returned no frames. "
          "Grant camera permission to Terminal/IDE and retry.")
    sys.exit(1)

print(f"Camera ready — {test_frame.shape[1]}x{test_frame.shape[0]}")
print(f"\nRunning Python (Ultralytics YOLO) inference benchmark ...")
print(f"Target : {NUM_FRAMES} frames at imgsz={INPUT_SIZE}")
print("Close the window or press 'q' to stop early.\n")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

frame_times    = []
frames_captured = 0

for i in range(NUM_FRAMES):
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Warning: failed to read frame {i}, stopping early")
        break
    frames_captured += 1

    # ── Timed inference ───────────────────────────────────────────────────
    start   = time.perf_counter()
    results = model(frame, imgsz=INPUT_SIZE, verbose=False)
    end     = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    frame_times.append(elapsed_ms)

    # ── Draw bounding boxes on the frame ──────────────────────────────────
    if results[0].boxes is not None and len(results[0].boxes):
        boxes   = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box
            colour = COLOURS[cls % len(COLOURS)]
            name   = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls{cls}"
            label  = f"{name} {conf:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # ── HUD overlay ───────────────────────────────────────────────────────
    avg_so_far = sum(frame_times) / len(frame_times)
    hud = (f"Frame {i+1}/{NUM_FRAMES}  |  "
           f"This: {elapsed_ms:.1f} ms  |  "
           f"Avg: {avg_so_far:.1f} ms  |  "
           f"FPS: {1000/avg_so_far:.1f}")
    cv2.putText(frame, hud, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)

    if (i + 1) % 10 == 0:
        print(f"  Frame {i+1:3d}/{NUM_FRAMES}  {elapsed_ms:7.2f} ms  "
              f"(avg {avg_so_far:.2f} ms)")

    # Allow early exit with 'q' or ESC
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        print("Early exit requested.")
        break

cap.release()
cv2.destroyAllWindows()

if not frame_times:
    print("ERROR: No frames were processed.")
    sys.exit(1)

# ── Stats ─────────────────────────────────────────────────────────────────────
avg_time = sum(frame_times) / len(frame_times)
fps      = 1000.0 / avg_time
min_time = min(frame_times)
max_time = max(frame_times)

print(f"\n=== Python Benchmark Results ({frames_captured} frames) ===")
print(f"  Avg inference time : {avg_time:.2f} ms")
print(f"  FPS                : {fps:.2f}")
print(f"  Min                : {min_time:.2f} ms")
print(f"  Max                : {max_time:.2f} ms")

results_data = {
    "backend": "Python (Ultralytics YOLO / PyTorch)",
    "model": "best.pt",
    "imgsz": INPUT_SIZE,
    "frames": frames_captured,
    "avg_inference_ms": round(avg_time, 4),
    "fps": round(fps, 4),
    "min_inference_ms": round(min_time, 4),
    "max_inference_ms": round(max_time, 4),
}

with open("benchmark_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

print("\nResults saved to benchmark_results.json")
