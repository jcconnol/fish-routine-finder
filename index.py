import cv2
import numpy as np

# Open the video
video_name = "video_2025-08-01_17-52-58"
cap = cv2.VideoCapture(f"videos/{video_name}.mp4")
# Works!
# PERCENT_DIFF = 0.03
# hex_color = "#daab1a"

# Works better?
PERCENT_DIFF = 0.02
hex_color = "#ffe400"

def hex_to_hsv(hex_color):
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgb_np = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)[0][0]
    return hsv

def get_hsv_range(hsv_color, percent=PERCENT_DIFF):
    h, s, v = hsv_color
    h_range = int(h * percent)
    s_range = int(s * percent)
    v_range = int(v * percent)

    lower = np.array([
        max(0, h - h_range),
        max(0, s - s_range),
        max(0, v - v_range)
    ])
    upper = np.array([
        min(179, h + h_range),
        min(255, s + s_range),
        min(255, v + v_range)
    ])
    return lower, upper


hsv_target = hex_to_hsv(hex_color)
lower_bound, upper_bound = get_hsv_range(hsv_target, percent=0.15)

# Track frame positions
frame_positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold to isolate goldfish-colored pixels
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find coordinates of matching pixels
    ys, xs = np.where(mask > 0)

    if len(xs) > 0 and len(ys) > 0:
        avg_x = int(np.mean(xs))
        avg_y = int(np.mean(ys))

        frame_positions.append((cap.get(cv2.CAP_PROP_POS_FRAMES), avg_x, avg_y))

        #cv2.circle(frame, (avg_x, avg_y), 5, (255, 0, 0), -1)

    # Show video (optional)
    # cv2.imshow("Tracking", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()

# Save positions to file (optional)
with open(f"findings/{video_name}.csv", "w") as f:
    for frame_num, x, y in frame_positions:
        f.write(f"{int(frame_num)},{x},{y}\n")
