"""Capture multiple local webcam samples and enroll the EDITH owner."""

from __future__ import annotations

import argparse
import time

from edith.owner_verification import OwnerVerifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", default=".edith-owner.json")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as error:
        raise SystemExit("Install OpenCV before enrollment.") from error

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}.")

    frames = []
    print("Look at the camera. Keep only your face in view.")
    print(f"Capturing up to {args.samples} samples; press Q to cancel.")
    try:
        while len(frames) < args.samples:
            ok, frame = camera.read()
            if not ok:
                continue
            cv2.imshow("EDITH owner enrollment (Q to cancel)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                raise SystemExit("Enrollment cancelled.")
            frames.append(frame.copy())
            print(f"Captured sample {len(frames)}/{args.samples}")
            time.sleep(max(0.0, args.interval))
    finally:
        camera.release()
        cv2.destroyAllWindows()

    verifier = OwnerVerifier(args.output)
    verifier.enroll_many(frames)
    print(f"Owner encoding saved locally to {args.output}")


if __name__ == "__main__":
    main()
