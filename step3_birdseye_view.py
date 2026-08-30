import cv2
import numpy as np
from config import BEV_DST_OFFSET_X

def process_birdseye_view(binary_mask, polygon_from_roi):
    height, width = binary_mask.shape[:2]
    
    pts = polygon_from_roi[0] 
    
    src_points = np.float32([
        [pts[0][0], pts[0][1]],   
        [pts[1][0], pts[1][1]],   
        [pts[2][0], pts[2][1]],   
        [pts[3][0], pts[3][1]]    
    ])
    
    dst_points = np.float32([
        [BEV_DST_OFFSET_X, height],     
        [BEV_DST_OFFSET_X, 0],          
        [width - BEV_DST_OFFSET_X, 0],          
        [width - BEV_DST_OFFSET_X, height]      
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points) 
    
    birdseye = cv2.warpPerspective(binary_mask, M, (width, height), flags=cv2.INTER_LINEAR)
    
    return birdseye, M, Minv