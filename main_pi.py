import cv2
import numpy as np
import time
import csv
import math
from threading import Thread
from queue import Queue

# IMPORT CẤU HÌNH CHUNG
from config import FRAME_WIDTH, FRAME_HEIGHT, BOTTOM_Y_PERCENTAGE

# IMPORT CÁC MODULE THUẬT TOÁN
from step1_preprocessing import process_frame
from step2_dynamic_roi import process_dynamic_roi
from step3_birdseye_view import process_birdseye_view 
from step4_5_lane_detection import LaneTracker, get_histogram_peaks, sliding_window, search_around_poly, fit_polynomial
from step6_display import measure_curvature_and_offset, draw_lane_and_hud


# ==========================================
# LUỒNG 1: ĐỌC VIDEO ĐẦU VÀO (PRODUCER)
# ==========================================
class ThreadedVideo:
    def __init__(self, path, queue_size=64):
        self.stream = cv2.VideoCapture(path)
        self.Q = Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
                    return
                self.Q.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        self.stream.release()


# ==========================================
# LUỒNG 3: GHI VIDEO ĐẦU RA (CONSUMER)
# ==========================================
class ThreadedWriter:
    def __init__(self, filename, fourcc, fps, frame_size, queue_size=64):
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.Q = Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.stopped and self.Q.empty():
                break
                
            if not self.Q.empty():
                frame = self.Q.get()
                self.writer.write(frame)
            else:
                time.sleep(0.01)
                
        self.writer.release()

    def write(self, frame):
        self.Q.put(frame)

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()


# ==========================================
# LUỒNG 2: HÀM CHẠY CHÍNH (MAIN THREAD - BỘ NÃO)
# ==========================================
def main():
    video_path = 'chuyenlan1.mp4' 
    
    print("==================================================")
    print("🚀 BẮT ĐẦU CHẠY ADAS TRÊN RASPBERRY PI (HEADLESS MODE)")
    print("✨ KIẾN TRÚC 3 LUỒNG + BỘ ĐẾM DUY TRÌ CẢNH BÁO")
    print("==================================================")
    
    # KÍCH HOẠT LUỒNG ĐỌC
    print("⏳ Đang khởi động Luồng 1 (Đọc video)...")
    cap = ThreadedVideo(video_path).start()
    time.sleep(1.0) 
    
    if not cap.more():
        print(f"❌ LỖI: Không thể mở file video '{video_path}'")
        return

    # KÍCH HOẠT LUỒNG GHI
    print("⏳ Đang khởi động Luồng 3 (Ghi thẻ nhớ)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_filename = 'pi_final_output.mp4'
    # Gắn cứng 30 FPS cho chuẩn video xuất ra trên Pi
    writer = ThreadedWriter(out_filename, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT)).start()

    # KHỞI TẠO CÁC BIẾN HỆ THỐNG
    prev_vp = None
    left_tracker = LaneTracker()
    right_tracker = LaneTracker()
    system_logs = []  
    frame_count = 0
    start_sim_time = time.time()

    # BIẾN QUẢN LÝ DUY TRÌ CẢNH BÁO (ĐỒNG BỘ 1:1 VỚI MAIN.PY)
    warning_hold_frames = 0
    last_warning_dir = "Unknown"

    print(f"🎬 Đang xử lý dữ liệu ở tốc độ cao nhất (Không mở giao diện đồ họa)...")
    print("==================================================")

    # VÒNG LẶP CHÍNH CỦA BỘ NÃO
    while cap.more():
        frame = cap.read()
        if frame is None: 
            break 
            
        frame_start_time = time.time() 
        frame_count += 1
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # --- ĐƯỜNG ỐNG THUẬT TOÁN ---
        final_mask, gamma_val, mean_bright, roi_rect = process_frame(frame)
        
        polygon, current_vp, lines, masked_edges = process_dynamic_roi(
            binary_mask=final_mask, prev_vp=prev_vp, 
            frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT,
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

        # PHÁT HIỆN CHUYỂN LÀN VỚI VÙNG ĐỆM MARGIN (ĐỒNG BỘ VỚI MAIN.PY)
        if left_fit_smooth is not None and right_fit_smooth is not None:
            l_bot_x = left_fit_smooth[0]*FRAME_HEIGHT**2 + left_fit_smooth[1]*FRAME_HEIGHT + left_fit_smooth[2]
            r_bot_x = right_fit_smooth[0]*FRAME_HEIGHT**2 + right_fit_smooth[1]*FRAME_HEIGHT + right_fit_smooth[2]
            
            center_x = FRAME_WIDTH / 2
            lane_width_px = r_bot_x - l_bot_x
            margin = lane_width_px * 0.4  
            
            if l_bot_x > (center_x + margin) or r_bot_x < (center_x - margin):
                left_tracker = LaneTracker()   
                right_tracker = LaneTracker()  

        # TÍNH TOÁN VẬT LÝ
        curvature, offset_m, direction, is_departure = measure_curvature_and_offset(left_fit_smooth, right_fit_smooth)
        
        # LOGIC KÉO DÀI THỜI GIAN CẢNH BÁO (WARNING HOLD)
        if is_departure:
            warning_hold_frames = 40
            last_warning_dir = direction
        elif warning_hold_frames > 0:
            is_departure = True
            direction = last_warning_dir
            warning_hold_frames -= 1

        # VẼ HUD (Đúng 9 tham số chuẩn của file step6_display.py)
        final_hud_img = draw_lane_and_hud(
            frame, warped_binary, left_fit_smooth, right_fit_smooth, 
            Minv, curvature, offset_m, direction, is_departure
        )

        # --- GHI KẾT QUẢ VÀO VIDEO VÀ RAM ---
        writer.write(final_hud_img)
        
        process_time_ms = (time.time() - frame_start_time) * 1000 
        instant_fps = 1000.0 / process_time_ms if process_time_ms > 0 else 0
        rad_val = curvature if curvature is not None else -1
        
        system_logs.append([frame_count, round(instant_fps, 1), round(rad_val, 1), round(offset_m, 2), direction])

        if frame_count % 30 == 0:
            rad_str = f"{curvature:.0f}m" if curvature is not None else "Unknown"
            status = "WARN" if is_departure else "OK"
            print(f"⏳ Frame {frame_count:04d} | Radius: {rad_str} | Offset: {offset_m:.2f}m {direction} | Status: {status} | CPU Time: {process_time_ms:.1f}ms")

    # ==========================================
    # DỌN DẸP TÀI NGUYÊN VÀ TỔNG KẾT
    # ==========================================
    print("==================================================")
    print("⏳ Đang dọn dẹp bộ nhớ và ghi Log xuống thẻ SD. Vui lòng đợi...")
    cap.stop()
    writer.stop() 
    
    # Xả dữ liệu từ RAM xuống file CSV
    csv_filename = "pi_adas_log.csv"
    with open(csv_filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Frame_ID', 'Instant_FPS', 'Radius_m', 'Offset_m', 'Direction'])
        csv_writer.writerows(system_logs)
    
    total_time = time.time() - start_sim_time
    avg_fps = frame_count / (total_time - 1.0) 
    
    print("🎉 QUÁ TRÌNH CHẠY TRÊN PI HOÀN TẤT THÀNH CÔNG!")
    print(f"📊 Tổng số khung hình: {frame_count} frames")
    print(f"⚡ TỐC ĐỘ FPS TRUNG BÌNH THỰC TẾ: {avg_fps:.1f} FPS")
    print(f"📁 Video lưu tại: {out_filename}")
    print(f"📝 File Log lưu tại: {csv_filename}")
    print("==================================================")

if __name__ == "__main__":
    main()