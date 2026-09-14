import cv2
import numpy as np
import time
import math

from config import FRAME_WIDTH, FRAME_HEIGHT, BOTTOM_Y_PERCENTAGE
from step1_preprocessing import process_frame
from step2_dynamic_roi import process_dynamic_roi
from step3_birdseye_view import process_birdseye_view 
from step4_5_lane_detection import LaneTracker, get_histogram_peaks, sliding_window, search_around_poly, fit_polynomial
from step6_display import measure_curvature_and_offset, draw_lane_and_hud

def main():
    cap = cv2.VideoCapture('test10.mp4') #thay đổi video input tại đây
    fps = cap.get(cv2.CAP_PROP_FPS) 
    if fps == 0 or math.isnan(fps): fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_video = cv2.VideoWriter('adas_final_output.mp4', fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    fps_history = []
    prev_vp = None
    view_mode = ord('9') 

    left_tracker = LaneTracker()
    right_tracker = LaneTracker()

    warning_hold_frames = 0
    last_warning_dir = "Unknown"

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        start_time = time.perf_counter()
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        final_mask, gamma_val, mean_bright, roi_rect = process_frame(frame)
        
        polygon, current_vp, lines, masked_edges = process_dynamic_roi(
            binary_mask=final_mask, 
            prev_vp=prev_vp, 
            frame_width=FRAME_WIDTH, 
            frame_height=FRAME_HEIGHT,
            bottom_y_percentage=BOTTOM_Y_PERCENTAGE
        )
        prev_vp = current_vp

        black_mask = np.zeros_like(final_mask)
        cv2.fillPoly(black_mask, [polygon], 255)
        clean_roi_mask = cv2.bitwise_and(final_mask, black_mask)
        warped_binary, M, Minv = process_birdseye_view(clean_roi_mask, polygon)

        use_sliding_window = True
        
        if left_tracker.best_fit is not None and right_tracker.best_fit is not None:
            leftx, lefty, rightx, righty, pixel_img = search_around_poly(
                warped_binary, left_tracker.best_fit, right_tracker.best_fit
            )
            if len(leftx) > 25 and len(rightx) > 25:
                use_sliding_window = False
            else:
                left_tracker = LaneTracker() 
                right_tracker = LaneTracker()

        if use_sliding_window:
            hist, leftx_base, rightx_base = get_histogram_peaks(warped_binary)
            leftx, lefty, rightx, righty, pixel_img = sliding_window(warped_binary, leftx_base, rightx_base)

        left_fit_smooth, right_fit_smooth, left_fitx, right_fitx, ploty, tracked_img = fit_polynomial(
            FRAME_HEIGHT, leftx, lefty, rightx, righty, pixel_img, left_tracker, right_tracker
        )

        if left_fit_smooth is not None and right_fit_smooth is not None:
            l_bot_x = left_fit_smooth[0]*FRAME_HEIGHT**2 + left_fit_smooth[1]*FRAME_HEIGHT + left_fit_smooth[2]
            r_bot_x = right_fit_smooth[0]*FRAME_HEIGHT**2 + right_fit_smooth[1]*FRAME_HEIGHT + right_fit_smooth[2]
            
            center_x = FRAME_WIDTH / 2
            lane_width_px = r_bot_x - l_bot_x
            margin = lane_width_px * 0.4  
            
            if l_bot_x > (center_x + margin) or r_bot_x < (center_x - margin):
                left_tracker = LaneTracker()   
                right_tracker = LaneTracker()  

        curvature, offset_m, direction, is_departure = measure_curvature_and_offset(left_fit_smooth, right_fit_smooth)
        
        if is_departure:
            warning_hold_frames = 40
            last_warning_dir = direction
        elif warning_hold_frames > 0:
            is_departure = True
            direction = last_warning_dir
            warning_hold_frames -= 1

        final_hud_img = draw_lane_and_hud(
            frame, warped_binary, left_fit_smooth, right_fit_smooth, 
            Minv, curvature, offset_m, direction, is_departure
        )

        display_frame = None

        if view_mode == ord('0'):
            display_frame = frame.copy()
            x1, y1, x2, y2 = roi_rect
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
            cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
            cv2.putText(display_frame, "0. Brightness Sensing ROI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif view_mode == ord('1'):
            display_frame = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(display_frame, "1. Binary Mask", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif view_mode == ord('2'):
            display_frame = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(display_frame, "2. Masked Canny Edges", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif view_mode == ord('3') or view_mode == ord('4'):
            display_frame = frame.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    color = (255, 0, 255) if view_mode == ord('3') else (0, 255, 0)
                    cv2.line(display_frame, (x1, y1), (x2, y2), color, 3) 
            if view_mode == ord('4') and current_vp is not None:
                cv2.circle(display_frame, current_vp, 10, (0, 255, 255), -1)
            title = "3. Hough Lines" if view_mode == ord('3') else "4. Vanishing Point"
            cv2.putText(display_frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif view_mode == ord('5'):
            display_frame = cv2.bitwise_and(frame, cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR))
            cv2.polylines(display_frame, [polygon], True, (0, 255, 0), 2) 
            cv2.putText(display_frame, "5. Dynamic ROI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif view_mode == ord('6'):
            display_frame = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR)
            cv2.putText(display_frame, "6. Bird's Eye View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif view_mode == ord('7'):
            display_frame = pixel_img
            cv2.putText(display_frame, "7. Pixel Extraction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif view_mode == ord('8'):
            display_frame = tracked_img
            cv2.putText(display_frame, "8. RANSAC + Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else: 
            display_frame = final_hud_img 
            cv2.putText(display_frame, "9. Final LDW Output", (20, FRAME_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        end_time = time.perf_counter()
        time_diff = end_time - start_time
        current_fps = 1.0 / time_diff if time_diff > 0 else 0
        
        fps_history.append(current_fps)
        if len(fps_history) > 10: 
            fps_history.pop(0)
        smoothed_fps = sum(fps_history) / len(fps_history)

        fps_text = f"FPS: {smoothed_fps:.1f}"
        
        (text_width, text_height), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        fps_x = FRAME_WIDTH - text_width - 20
        fps_y = 40
        
        cv2.putText(final_hud_img, fps_text, (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(final_hud_img, fps_text, (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        info_text = f"Luma: {mean_bright:.1f} | Gamma: {gamma_val:.2f}"
        cv2.putText(final_hud_img, info_text, (20, FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(final_hud_img, info_text, (20, FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("ADAS Project", display_frame)
        
        #ghi vào video
        out_video.write(display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): 
            break
        elif key in [ord(str(i)) for i in range(10)]: 
            view_mode = key 
        elif key == ord('s') or key == ord('S'):
            filename = f"screenshot_mode_{chr(view_mode)}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"File duoc luu: {filename}")

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    
    print("Luu video vao file: adas_final_output.mp4")

if __name__ == "__main__":
    main()