import cv2
import numpy as np
import time

previous_offset = 0

# -----------------------------
# REGION OF INTEREST FUNCTION
# -----------------------------
def region_of_interest(image):

    height = image.shape[0]
    width = image.shape[1]

    mask = np.zeros_like(image)

    polygon = np.array([[
    (0, height),
    (width*1.15, height),
    (int(width*0.8), int(height*0.15)),
    (int(width*0.1), int(height*0.15))
]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# -----------------------------
# Slope detection
# -----------------------------
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return np.array([])

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Ignore horizontal lines
        if abs(slope) < 0.2:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    averaged_lines = []

    if len(left_fit) > 0:
        left_avg = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_avg)
        averaged_lines.append(left_line)

    if len(right_fit) > 0:
        right_avg = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_avg)
        averaged_lines.append(right_line)

    return np.array(averaged_lines)

# -----------------------------
# coordinate system
# -----------------------------

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters

    y1 = image.shape[0]
    y2 = int(y1 * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

# ----------------------
# steering 
# ----------------------

def compute_steering(frame, lines):

    if len(lines) < 2:
        return "No Lane Detected", frame.shape[1] // 2

    left_line = lines[0]
    right_line = lines[1]

    _, _, left_x2, _ = left_line
    _, _, right_x2, _ = right_line

    lane_center = (left_x2 + right_x2) // 2

    frame_center = frame.shape[1] // 2

    global previous_offset

    current_offset = lane_center - frame_center

    offset = int(0.8 * previous_offset + 0.2 * current_offset)

    previous_offset = offset

    if offset < -50:
        return "Steer Left", lane_center

    elif offset > 50:
        return "Steer Right", lane_center

    else:
        return "Go Straight", lane_center


# -----------------------------
# DISPLAY LINES FUNCTION
# -----------------------------
def display_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 10)

    return line_image


# -----------------------------
# MAIN PROGRAM
# -----------------------------

# Read image
cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('test_video1.mp4')


while cap.isOpened():

    ret, frame = cap.read()
    start = time.time()
    if not ret:
        break

# Check image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 150, 250)

    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        2,
        np.pi/180,
        50,
        np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

    averaged_lines = average_slope_intercept(frame, lines)
    
    steering, lane_center = compute_steering(frame, averaged_lines)
    print(steering)

    line_image = display_lines(frame, averaged_lines)

    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    frame_center = frame.shape[1] // 2

    cv2.circle(
     combo,
     (frame_center, frame.shape[0] - 50),
     8,
     (255, 0, 0),
     -1
    )  

    cv2.circle(
    combo,
    (lane_center, frame.shape[0] - 100),
    8,
    (0, 255, 255),
    -1
    )  

    cv2.line(
    combo,
    (frame_center, frame.shape[0] - 50),
    (lane_center, frame.shape[0] - 100),
    (0, 0, 255),
    4
    )

    cv2.putText(
     combo,
     steering,
     (50, 50),
     cv2.FONT_HERSHEY_SIMPLEX,
     1,
     (0, 0, 255),
     2  
    )
    cv2.imshow("Lane Detection", combo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release(1)
cv2.destroyAllWindows()
    