import cv2
import numpy as np

def auto_adjust_gamma(image, min_gamma=0.05, max_gamma=2.0):
    height, width = image.shape[:2]
    
    x_start = int(width * 0.1)
    x_end = int(width * 0.9)

    y_start, y_end = int(height * 0.4), height 
    road_roi = image[y_start:y_end, x_start:x_end]
    
    gray_roi = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
    mean_brightness_road = np.mean(gray_roi)
      
    gamma = np.interp(mean_brightness_road, [20, 60, 100, 140, 180], [0.05, 0.15, 0.4, 1.00, 2.00])
    #gamma = np.interp(mean_brightness_road, [20, 60, 100, 140, 180], [0.4, 0.8, 1.0, 1.5, 2.5])
    gamma = np.clip(gamma, min_gamma, max_gamma)
    
    table = np.array([((i / 255.0) ** gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    corrected = cv2.LUT(image, table)
    
    roi_rect = (x_start, y_start, x_end, y_end)
    
    return corrected, gamma, mean_brightness_road, roi_rect

def process_frame(frame, block_size=61, C_value=-15):
    frame_gamma, gamma_value, mean_bright, roi_rect = auto_adjust_gamma(frame)
    
    lab = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    
    binary = cv2.adaptiveThreshold(
        L_clahe, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        C_value
    )

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    horizontal_noise = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)
    
    binary_no_horizontal = cv2.subtract(binary, horizontal_noise)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    binary_cleaned = cv2.morphologyEx(binary_no_horizontal, cv2.MORPH_OPEN, kernel_vertical)
    
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_final = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, kernel_smooth)
    
    return binary, gamma_value, mean_bright, roi_rect