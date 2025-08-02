import cv2
import numpy as np
import os
import boto3
from botocore.exceptions import ClientError

"""
for each video in bucket folder - done
   if not in s3 csv list - done
       download file from s3 - 
       find coordinates of certain color in each frame of a video - done
       save to csv - done
       upload csv to s3
       delete local mp4
       delete local csv
"""

# Bucket things:
BUCKET_NAME = 'fish-cam'
FILE_PREFIX = "videos/video"

s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')

pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=FILE_PREFIX)

# Video things:
PERCENT_DIFF = 0.02
HEX_COLOR = "#ffe400"

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

def check_s3_object_exists(bucket_name, object_key):
    s3_client = boto3.client('s3')
    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            print(f"An error occurred: {e}")
            return False

def process_video(file_name):
    cap = cv2.VideoCapture(f"videos/{file_name}.mp4")
    
    hsv_target = hex_to_hsv(HEX_COLOR)
    lower_bound, upper_bound = get_hsv_range(hsv_target, percent=0.15)

    frame_positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        ys, xs = np.where(mask > 0)

        if len(xs) > 0 and len(ys) > 0:
            avg_x = int(np.mean(xs))
            avg_y = int(np.mean(ys))

            frame_positions.append((cap.get(cv2.CAP_PROP_POS_FRAMES), avg_x, avg_y))

            # optional
            cv2.circle(frame, (avg_x, avg_y), 5, (255, 0, 0), -1)

        # Show video (optional)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save positions to file (optional)
    with open(f"csvs/{file_name}.csv", "w") as f:
        for frame_num, x, y in frame_positions:
            f.write(f"{int(frame_num)},{x},{y}\n")

def download_video_from_s3(mp4_file_path_name):
    s3.download_file(BUCKET_NAME, mp4_file_path_name, mp4_file_path_name)

def upload_csv_to_s3(csv_file_path_name):
    s3.upload_file(csv_file_path_name, BUCKET_NAME, csv_file_path_name)

def delete_local_csv(csv_file_path_name):
    os.remove(csv_file_path_name)

def delete_local_mp4(mp4_file_path_name):
    os.remove(mp4_file_path_name)

iter = 0
for page in pages:
    for obj in page['Contents']:
        mp4_file_path_name = obj['Key']
        mp4_file_name = mp4_file_path_name.split("/",1)[1]
        csv_file_name = mp4_file_name.replace(".mp4", ".csv")
        csv_file_path_name = f'csvs/{csv_file_name}'
        raw_file_name = mp4_file_name.replace(".mp4", "")
        
        if not check_s3_object_exists(BUCKET_NAME, csv_file_path_name):
            print(f"Object '{csv_file_path_name}' does not exist in bucket '{BUCKET_NAME}'.")
            download_video_from_s3(mp4_file_path_name)
            process_video(raw_file_name)
            upload_csv_to_s3(csv_file_path_name)
            delete_local_csv(csv_file_path_name)
            delete_local_mp4(mp4_file_path_name)
            
        iter = iter + 1
        if iter > 100:
            exit()
        

