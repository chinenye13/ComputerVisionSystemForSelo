
import os
import json
import cv2 as cv

# Files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAL_FILE = os.path.join(BASE_DIR, "calibration.json")
REF_FILE = os.path.join(BASE_DIR, "reference.json")

# Settings
CALIBRATION_MM = 85.6
TOLERANCE_MM = 1.0
SAMPLES_NEEDED = 30
MIN_AREA = 500


def length_px(contour):
    _, (w, h), _ = cv.minAreaRect(contour)
    return max(w, h)


def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)
    return None


def save_json(path, data):
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


# Load saved measurements
calibration = load_json(CAL_FILE)
reference = load_json(REF_FILE)

# Camera
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

mode = None
samples = []

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_size = [frame.shape[1], frame.shape[0]]

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    otsu, mask = cv.threshold(
        blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    if cv.countNonZero(mask) > mask.size / 2:
        mask = cv.bitwise_not(mask)

    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    # Find and measure chunks

    biggest = None

    for chunk in contours:
        area = cv.contourArea(chunk)

        if area < MIN_AREA:
            continue

        if biggest is None or area > cv.contourArea(biggest):
            biggest = chunk

        x, y, w, h = cv.boundingRect(chunk)
        length = length_px(chunk)

        color = (255, 255, 0)
        label = f"{length:.0f}px"

        if calibration:
            length_mm = length * calibration["mm_per_px"]
            label = f"{length_mm:.1f}mm"

            if reference and mode is None:
                target = reference["length_mm"]
                tolerance = reference["tolerance_mm"]

                if target - tolerance <= length_mm <= target + tolerance:
                    color = (0, 255, 0)
                    label = "OK " + label
                else:
                    color = (0, 0, 255)
                    label = "BAD " + label

        cv.drawContours(frame, [chunk], -1, color, 2)

        cv.putText(
            frame, label, (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # Calibration and teaching

    if mode:

        if biggest is not None:
            samples.append(length_px(biggest))

        cv.putText(
            frame,
            f"{mode}... {len(samples)}/{SAMPLES_NEEDED}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if len(samples) >= SAMPLES_NEEDED:

            average_px = sum(samples) / len(samples)

            if mode == "calibrating":

                calibration = {
                    "mm_per_px": CALIBRATION_MM / average_px,
                    "resolution": frame_size
                }

                save_json(CAL_FILE, calibration)

                print("Calibration saved!")

            elif mode == "teaching":

                reference = {
                    "length_mm": average_px * calibration["mm_per_px"],
                    "tolerance_mm": TOLERANCE_MM
                }

                save_json(REF_FILE, reference)

                print(
                    f"Reference saved: {reference['length_mm']:.1f}mm"
                )

            mode = None
            samples = []

    # Display instructions

    elif not calibration:
        cv.putText(
            frame, "Press K to calibrate",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 0), 2
        )

    elif not reference:
        cv.putText(
            frame, "Press S to teach reference chunk",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 0), 2
        )

    else:
        cv.putText(
            frame,
            f"Target: {reference['length_mm']:.1f}mm",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    # Display
    cv.imshow("Object Detection", frame)
    cv.imshow("Mask", mask)

    key = cv.waitKey(1)

    if key == ord("q"):
        break

    elif key == ord("k") and biggest is not None and mode is None:
        samples = []
        mode = "calibrating"

    elif key == ord("s") and biggest is not None and mode is None:
        if calibration:
            samples = []
            mode = "teaching"
        else:
            print("Calibrate first!")


cap.release()
cv.destroyAllWindows()
