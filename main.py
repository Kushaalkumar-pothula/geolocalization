import cv2
import numpy as np
import csv
from preprocess import preprocess

image = "0a487bfbbf0144d3a72d7522adc5cf8e"


def read_metadata(csv_file_path, target_image_id):
    found = False
    if not target_image_id.endswith('.png'):
        target_image_id += '.png'
        
    with open(csv_file_path, mode='r') as file:
        for row in csv.DictReader(file):
            if row['filename'] == f'{target_image_id}':
                return row['cam_k1'], row['cam_k2'], row['cam_k3'], row['cam_p1'], row['cam_p2'], row['alt'], row['yaw'], row['pitch'], row['roll']
                
        if not found:
            print(f"Image ID '{target_image_id}' not found in the CSV.")



if __name__ == "__main__":
    image_path = f"./dataset/d1/images/{image}.png"
    csv_path = f"./dataset/d1/images/metadata.csv"
    k1, k2, k3, p1, p2, alt_takeoff, yaw, pitch, roll = read_metadata(csv_path, image)
    print("k1: ", k1, "k2: ", k2, "k3: ", k3, "alt_takeoff: ", alt_takeoff, "yaw: ", yaw, "pitch: ", pitch, "roll: ", roll)
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = preprocess(raw, k1, k2, k3, p1, p2, yaw, pitch, roll, alt_takeoff)
    cv2.imshow('undistorted preprocessed', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()