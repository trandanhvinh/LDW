import cv2
import numpy as np
from config import *

class LaneTracker:
    def __init__(self, smooth_factor=TRACKING_SMOOTH_FACTOR):
        self.recent_fits = [] 
        self.smooth_factor = smooth_factor
        self.best_fit = None 

    def update(self, new_fit):
        if new_fit is None:
            return self.best_fit
            
        if self.best_fit is not None:
            diff_A = abs(new_fit[0] - self.best_fit[0])
            if diff_A > TRACKING_MAX_A_DIFF:
                return self.best_fit
                
        self.recent_fits.append(new_fit)        
        if len(self.recent_fits) > self.smooth_factor:
            self.recent_fits.pop(0)            
        self.best_fit = np.mean(self.recent_fits, axis=0)
        return self.best_fit
    
def get_histogram_peaks(binary_warped):
    height = binary_warped.shape[0] 
    bottom_half = binary_warped[height // 2 :, :]
    histogram = np.sum(bottom_half, axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return histogram, leftx_base, rightx_base

def sliding_window(binary_warped, leftx_base, rightx_base):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    nwindows = SLIDING_WINDOW_N
    margin = SLIDING_WINDOW_MARGIN
    minpix = SLIDING_WINDOW_MINPIX
    
    window_height = int(binary_warped.shape[0] // nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []
    
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix: 
            new_leftx = int(np.mean(nonzerox[good_left_inds]))
            #cập nhật nếu tâm không bị giật ngang quá gắt
            if abs(new_leftx - leftx_current) <= margin * 1.5:
                leftx_current = new_leftx

        if len(good_right_inds) > minpix: 
            new_rightx = int(np.mean(nonzerox[good_right_inds]))
            if abs(new_rightx - rightx_current) <= margin * 1.5:
                rightx_current = new_rightx
            
    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([])
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([])
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]
    
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img

def search_around_poly(binary_warped, left_fit, right_fit):
    margin = SEARCH_AROUND_MARGIN
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_fitx = left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2]
    right_fitx = right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]
    
    left_lane_inds = ((nonzerox > (left_fitx - margin)) & (nonzerox < (left_fitx + margin)))
    right_lane_inds = ((nonzerox > (right_fitx - margin)) & (nonzerox < (right_fitx + margin)))
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    return leftx, lefty, rightx, righty, out_img

def ransac_polyfit(x, y, max_iters=RANSAC_MAX_ITERS, threshold=RANSAC_THRESHOLD):
    if len(x) < RANSAC_MIN_POINTS:
        return None, None
        
    best_fit = None
    max_inliers = 0
    
    #Lấy mẫu thưa 3 để tăng tốc độ
    step = 3
    eval_x = x[::step]
    eval_y = y[::step]
    
    for _ in range(max_iters):
        idx = np.random.choice(len(x), 3, replace=False)
        x1, x2, x3 = x[idx]
        y1, y2, y3 = y[idx]
        
        y1, y2, y3 = float(y1), float(y2), float(y3)
        x1, x2, x3 = float(x1), float(x2), float(x3)
        denom = (y1 - y2) * (y1 - y3) * (y2 - y3)
        if abs(denom) < 1e-5:
            continue
        A = (x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2)) / denom
        
        if abs(A) > RANSAC_MAX_A_COEFF: 
            continue
            
        B = -(x1 * (y2**2 - y3**2) - x2 * (y1**2 - y3**2) + x3 * (y1**2 - y2**2)) / denom
        C = (x1 * y2 * y3 * (y2 - y3) - x2 * y1 * y3 * (y1 - y3) + x3 * y1 * y2 * (y1 - y2)) / denom

        fit = [A, B, C]
        
        fit_eval_x = fit[0]*eval_y**2 + fit[1]*eval_y + fit[2]
        errors = np.abs(eval_x - fit_eval_x)
        
        inliers_count = np.sum(errors < threshold)
        
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_fit = fit
            
    if best_fit is not None:
        final_fit_x = best_fit[0]*y**2 + best_fit[1]*y + best_fit[2]
        best_inlier_idx = np.abs(x - final_fit_x) < threshold
        
        if np.sum(best_inlier_idx) > 3:
            final_fit = np.polyfit(y[best_inlier_idx], x[best_inlier_idx], 2)
            return final_fit, best_inlier_idx
            
    return None, None

def fit_polynomial(img_height, leftx, lefty, rightx, righty, out_img, left_tracker, right_tracker):
    left_fitx, right_fitx = None, None
    ploty = np.linspace(0, img_height - 1, img_height)

    raw_left_fit, left_inliers = ransac_polyfit(leftx, lefty)
    smoothed_left_fit = left_tracker.update(raw_left_fit)
    
    if smoothed_left_fit is not None:
        left_fitx = smoothed_left_fit[0]*ploty**2 + smoothed_left_fit[1]*ploty + smoothed_left_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
        cv2.polylines(out_img, pts_left, False, (0, 255, 255), 4)
        if raw_left_fit is not None:
            outliers_idx = ~left_inliers
            out_img[lefty[outliers_idx], leftx[outliers_idx]] = [150, 150, 150] # Tô xám pixel nhiễu
            
    raw_right_fit, right_inliers = ransac_polyfit(rightx, righty)
    smoothed_right_fit = right_tracker.update(raw_right_fit)
    
    if smoothed_right_fit is not None:
        right_fitx = smoothed_right_fit[0]*ploty**2 + smoothed_right_fit[1]*ploty + smoothed_right_fit[2]
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)
        cv2.polylines(out_img, pts_right, False, (0, 255, 255), 4)
        
        if raw_right_fit is not None:
            outliers_idx = ~right_inliers
            out_img[righty[outliers_idx], rightx[outliers_idx]] = [150, 150, 150] # Tô xám pixel nhiễu

    return smoothed_left_fit, smoothed_right_fit, left_fitx, right_fitx, ploty, out_img