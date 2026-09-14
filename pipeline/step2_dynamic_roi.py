import cv2
import numpy as np
import math

from config import *

#Lấy mask động để tìm đường thẳng Hough
def _get_base_mask(width, height, bottom_y, vp=None):
    mask = np.zeros((height, width), dtype=np.uint8)
    horizon_y = int(height * BASE_MASK_HORIZON_RATIO) 
    vp_x = vp[0] if vp is not None else width // 2
        
    polygon = np.array([[
        (int(width * BASE_MASK_BOT_LEFT_RATIO), bottom_y),            
        (vp_x - DYNAMIC_BASE_MASK_TOP_HALF_WIDTH, horizon_y),         
        (vp_x + DYNAMIC_BASE_MASK_TOP_HALF_WIDTH, horizon_y),         
        (int(width * BASE_MASK_BOT_RIGHT_RATIO), bottom_y)            
    ]], np.int32)
    
    cv2.fillPoly(mask, [polygon], 255)
    return mask

#Phân loại đường thẳng trái/phải
def _classify(lines, width):
    left, right = [], []
    if lines is None: return left, right
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2: continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        if abs(m) < MIN_SLOPE or abs(m) > MAX_SLOPE: continue
        
        mid_x = (x1 + x2) / 2
        length_sq = (x2 - x1)**2 + (y2 - y1)**2 
        
        if m < 0 and mid_x < width * LEFT_LANE_MAX_X_RATIO: 
            left.append((length_sq, m, b))
        elif m > 0 and mid_x > width * RIGHT_LANE_MIN_X_RATIO:
            right.append((length_sq, m, b))
            
    left.sort(key=lambda x: x[0], reverse=True)
    right.sort(key=lambda x: x[0], reverse=True)
    
    left = [(m, b) for _, m, b in left[:10]]
    right = [(m, b) for _, m, b in right[:10]]
    
    return left, right

#Tìm điểm tụ
def _find_vanishing_point(left_lines, right_lines, frame_width, frame_height):
    intersections_x, intersections_y = [], []
    for left_m, left_b in left_lines:
        for right_m, right_b in right_lines:
            if left_m == right_m: continue
            x = (right_b - left_b) / (left_m - right_m)
            y = left_m * x + left_b
            if 0 <= x <= frame_width and 0 <= y <= frame_height:
                intersections_x.append(x)
                intersections_y.append(y)

    if len(intersections_x) > 0:
        vp_x = int(np.median(intersections_x))
        vp_y = int(np.median(intersections_y))
        return (vp_x, vp_y)
    return None    

#Tìm và vẽ Dynamic ROI
def _get_roi_polygon(width, height, vp, bottom_y):
    if vp is None:
        vp_y = int(height * FALLBACK_VP_Y_RATIO) 
        vp_x = width // 2
    else:
        vp_x, vp_y = vp
        
    top_y = min(max(vp_y + ROI_OFFSET_Y, 0), height)

    polygon = np.array([[
        (vp_x - ROI_BOTTOM_OFFSET_LEFT, bottom_y), 
        (vp_x - ROI_TOP_HALF_WIDTH, top_y),          
        (vp_x + ROI_TOP_HALF_WIDTH, top_y),          
        (vp_x + ROI_BOTTOM_OFFSET_RIGHT, bottom_y)  
    ]], np.int32)
    return polygon

def process_dynamic_roi(binary_mask, prev_vp, frame_width, frame_height, bottom_y_percentage=BOTTOM_Y_PERCENTAGE):
    bottom_y = int(frame_height * bottom_y_percentage)
    base_mask = _get_base_mask(frame_width, frame_height, bottom_y, vp=prev_vp)
    
    masked_binary = cv2.bitwise_and(binary_mask, base_mask)
    masked_edges = cv2.Canny(masked_binary, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
    
    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi/180, 
        threshold=HOUGH_THRESHOLD, 
        minLineLength=HOUGH_MIN_LINE_LEN, 
        maxLineGap=HOUGH_MAX_LINE_GAP
    )
    
    left_lines, right_lines = _classify(lines, frame_width)
    vp = _find_vanishing_point(left_lines, right_lines, frame_width, frame_height)
    
    if vp is not None:
        if prev_vp is None: 
            pass
        else:
            dist = math.hypot(vp[0] - prev_vp[0], vp[1] - prev_vp[1])
            alpha = min(VP_SMOOTH_MAX_ALPHA, max(VP_SMOOTH_MIN_ALPHA, dist / VP_SMOOTH_DIST_SCALE))
            vp = (int(alpha * vp[0] + (1 - alpha) * prev_vp[0]), 
                  int(alpha * vp[1] + (1 - alpha) * prev_vp[1]))
    else:
        if prev_vp is not None: 
            vp = prev_vp
            
    polygon = _get_roi_polygon(frame_width, frame_height, vp, bottom_y)
    
    return polygon, vp, lines, masked_edges