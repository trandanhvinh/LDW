import cv2
import numpy as np

from config import *

_ploty_trunc = None
_ploty_trunc_sq = None

def measure_curvature_and_offset(left_fit, right_fit, car_center_px=None):
    y_eval = FRAME_HEIGHT
    if car_center_px is None:
        car_center_px = FRAME_WIDTH / 2

    is_departure = False
    direction = "Unknown"
    offset_m = 0.0
    curvature = None

    left_x_bottom = None
    right_x_bottom = None

    if left_fit is not None:
        left_x_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    if right_fit is not None:
        right_x_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]


    #Check lấn làn trái
    if left_x_bottom is not None and car_center_px <= left_x_bottom:
        is_departure = True
        direction = "Left"
        offset_m = (left_x_bottom - car_center_px) * XM_PER_PIX 

    #Check lấn làn phải    
    elif right_x_bottom is not None and car_center_px >= right_x_bottom:
        is_departure = True
        direction = "Right"
        offset_m = (car_center_px - right_x_bottom) * XM_PER_PIX
        
    #an toàn
    elif left_x_bottom is not None and right_x_bottom is not None:
        direction = "Inside Lane"
        lane_center_px = (left_x_bottom + right_x_bottom) / 2
        offset_m = abs(car_center_px - lane_center_px) * XM_PER_PIX
        
        left_a_m = left_fit[0] * XM_PER_PIX / (YM_PER_PIX ** 2)
        left_b_m = left_fit[1] * XM_PER_PIX / YM_PER_PIX
        right_a_m = right_fit[0] * XM_PER_PIX / (YM_PER_PIX ** 2)
        right_b_m = right_fit[1] * XM_PER_PIX / YM_PER_PIX

        left_curverad = ((1 + (2 * left_a_m * (y_eval * YM_PER_PIX) + left_b_m) ** 2) ** 1.5) / np.absolute(2 * left_a_m)
        right_curverad = ((1 + (2 * right_a_m * (y_eval * YM_PER_PIX) + right_b_m) ** 2) ** 1.5) / np.absolute(2 * right_a_m)
        curvature = (left_curverad + right_curverad) / 2
    else:
        direction = "Unknown"

    return curvature, offset_m, direction, is_departure

def draw_lane_and_hud(frame, warped_binary, left_fit_smooth, right_fit_smooth, Minv, curvature, offset_m, direction, is_departure):
    global _ploty_trunc, _ploty_trunc_sq
    result = frame.copy()

    if is_departure: 
        warn_text = f"WARNING: DEPARTURE {direction.upper()}!"
        cv2.putText(result, warn_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(result, warn_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    #check mất vạch
    if left_fit_smooth is None and right_fit_smooth is None:
        if not is_departure:
            cv2.putText(result, "WARNING: LANE LOST!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result 
        
    if left_fit_smooth is None or right_fit_smooth is None:
        if not is_departure:
            cv2.putText(result, "POOR VISIBILITY - 1 LANE LOST", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    #vẽ vùng làn đường
    if left_fit_smooth is not None and right_fit_smooth is not None:
        height, width = warped_binary.shape
        start_y = int(height * 0.45)
        expected_len = height - start_y
        
        if _ploty_trunc is None or len(_ploty_trunc) != expected_len:
            _ploty_trunc = np.linspace(start_y, height - 1, expected_len)
            _ploty_trunc_sq = _ploty_trunc ** 2

        lane_color = (0, 0, 255) if is_departure else (0, 255, 0)
        
        left_fitx = left_fit_smooth[0]*_ploty_trunc_sq + left_fit_smooth[1]*_ploty_trunc + left_fit_smooth[2]
        right_fitx = right_fit_smooth[0]*_ploty_trunc_sq + right_fit_smooth[1]*_ploty_trunc + right_fit_smooth[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, _ploty_trunc]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, _ploty_trunc])))])
        pts = np.hstack((pts_left, pts_right)).astype(np.float32)

        warped_pts = cv2.perspectiveTransform(pts, Minv).astype(np.int32)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [warped_pts[0]], lane_color)
        result = cv2.addWeighted(result, 1, overlay, 0.3, 0)

    #hiển thị thông số
    if curvature is not None:
        if curvature > 3000: curv_text = "Radius: Straight"
        else: curv_text = f"Radius: {curvature:.0f} m"
    else:
        curv_text = "Radius: Unknown"

    if is_departure: offset_text = f"Over Lane: {offset_m:.2f} m {direction}"
    else: offset_text = f"Offset: {offset_m:.2f} m {direction}"

    cv2.putText(result, curv_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(result, curv_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(result, offset_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(result, offset_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return result