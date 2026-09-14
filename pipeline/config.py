FRAME_WIDTH = 640
FRAME_HEIGHT = 360

BOTTOM_Y_PERCENTAGE = 0.95        #đáy của ROI

#Base mask
BASE_MASK_HORIZON_RATIO = 0.6
BASE_MASK_BOT_LEFT_RATIO = 0.14
BASE_MASK_BOT_RIGHT_RATIO = 0.92
DYNAMIC_BASE_MASK_TOP_HALF_WIDTH = 43

#Lọc canny và hough
CANNY_LOW_THRESH = 50             # Ngưỡng dưới của Canny
CANNY_HIGH_THRESH = 150           # Ngưỡng trên của Canny
HOUGH_THRESHOLD = 40              # Số điểm ảnh tối thiểu để tạo thành đường thẳng
HOUGH_MIN_LINE_LEN = 30           # Chiều dài tối thiểu của một vạch kẻ đường
HOUGH_MAX_LINE_GAP = 40           # Khoảng cách tối đa để nối 2 đoạn vạch đứt

#Lọc và làm mượt điểm tụ
MIN_SLOPE = 0.5                   # Độ dốc tối thiểu để không nhận diện nhầm vạch ngang
MAX_SLOPE = 2.5                   # Độ dốc tối đa để không nhận diện nhầm lề dọc
LEFT_LANE_MAX_X_RATIO = 0.65      # Giới hạn vùng tìm vạch trái
RIGHT_LANE_MIN_X_RATIO = 0.35     # Giới hạn vùng tìm vạch phải
VP_SMOOTH_MAX_ALPHA = 0.8         # Hệ số bám đuổi tối đa (khi điểm tụ dịch chuyển mạnh)
VP_SMOOTH_MIN_ALPHA = 0.05        # Hệ số bám đuổi tối thiểu (khi xe chạy ổn định)
VP_SMOOTH_DIST_SCALE = 200.0      # Thang đo khoảng cách để điều chỉnh alpha

#Dynamic ROI
FALLBACK_VP_Y_RATIO = 0.80        # Vị trí điểm tụ giả định nếu mất dấu (80% chiều cao)
ROI_OFFSET_Y = 25
ROI_TOP_HALF_WIDTH = 45
ROI_BOTTOM_OFFSET_LEFT = 210
ROI_BOTTOM_OFFSET_RIGHT = 210
BEV_DST_OFFSET_X = 200

#Sliding Window
SLIDING_WINDOW_N = 5              # Số lượng cửa sổ xếp chồng lên nhau
SLIDING_WINDOW_MARGIN = 30        # Khoảng cách từ tâm cửa sổ ra mỗi bên
SLIDING_WINDOW_MINPIX = 25        # Số điểm sáng tối thiểu để cập nhật lại tâm cửa sổ mới
SEARCH_AROUND_MARGIN = 40         # Phạm vi tìm kiếm pixel bao quanh vạch đường của khung hình trước

#RANSAC
RANSAC_MAX_ITERS = 10         # Số vòng lặp lấy mẫu ngẫu nhiên để tìm đường cong chuẩn nhất
RANSAC_THRESHOLD = 12.0           # Sai số tối đa (pixel) để 1 điểm được coi là thuộc vạch kẻ đường
RANSAC_MIN_POINTS = 10            # Số lượng pixel tối thiểu lọt vào cửa sổ thì mới nội suy phương trình
RANSAC_MAX_A_COEFF = 0.001        # Hệ số cong A tối đa cho phép để loại bỏ các đường cong quá cong

#Tracking
TRACKING_SMOOTH_FACTOR = 7        # Số khung hình ghi nhớ trong bộ đệm để lấy trung bình cộng
TRACKING_MAX_A_DIFF = 0.001       # Độ lệch độ cong A tối đa cho phép giữa 2 khung hình liên tiếp để chống giật


#Quy đổi pixel sang mét
YM_PER_PIX = 30.0 / FRAME_HEIGHT  # khung hình camera nhìn được đoạn đường dài 30m
XM_PER_PIX = 3.7 / 700.0          # Chuẩn làn đường quốc tế rộng 3.7m -> tương đương 700 pixel trên BEV

LDW_WARNING_THRESHOLD = 0.4       # Ngưỡng lệch tâm an toàn (mét). Quá 0.4m sẽ báo động đỏ