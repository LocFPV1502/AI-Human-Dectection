import cv2
import cvlib as cv
import numpy as np
from cvlib.object_detection import detect_common_objects, draw_bbox
import time
import multiprocessing as mp # Import multiprocessing

# --- CẤU HÌNH ---
camera_index = 'http://192.168.0.106:8160/stream.mjpg'# Webcam mặc định của laptop
confidence_threshold = 0.8 # Ngưỡng tin cậy để phát hiện vật thể
# -----------------

def frame_grabber(queue, camera_idx):
    """Tiến trình đọc khung hình từ webcam và đưa vào hàng đợi."""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Lỗi Frame Grabber: Không thể mở webcam với index {camera_idx}.")
        queue.put(None) # Đặt None vào hàng đợi để báo hiệu lỗi
        return

    print("Frame Grabber: Webcam đã được khởi tạo.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame Grabber: Lỗi đọc khung hình hoặc webcam bị ngắt kết nối.")
            break
        
        # Đặt khung hình vào hàng đợi (có thể giới hạn kích thước hàng đợi để tránh tràn bộ nhớ)
        if queue.qsize() < 10: # Giới hạn hàng đợi để không bị chậm trễ quá nhiều
            queue.put(frame)
        else:
            # print("Frame Grabber: Hàng đợi đầy, bỏ qua khung hình.")
            pass # Bỏ qua khung hình nếu hàng đợi đầy để không bị tắc nghẽn

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    print("Frame Grabber: Đã thoát.")
    queue.put(None) # Đặt None vào hàng đợi để báo hiệu kết thúc
def pid_controller(error_queue, Kp, Ki, Kd):
    """Tiến trình tính toán giá trị điều khiển PID."""
    print("PID Controller: Bắt đầu hoạt động.")

    # Các biến cho bộ điều khiển PID
    last_error = 0
    integral = 0
    neutral_throttle = 1500 # Giá trị ga trung bình khi không có sai số

    while True:
        # Lấy sai số từ hàng đợi
        error = error_queue.get()
        if error is None: # Tín hiệu kết thúc
            break

        # --- TÍNH TOÁN PID ---
        # Thành phần Tỷ lệ (Proportional)
        P_term = Kp * error

        # Thành phần Tích phân (Integral)
        integral += error
        I_term = Ki * integral
        
        # Chống "Integral Windup" (khi tích phân quá lớn)
        # Giới hạn giá trị của thành phần tích phân để tránh vọt lố
        if I_term > 200: I_term = 200
        if I_term < -200: I_term = -200

        # Thành phần Vi phân (Derivative)
        derivative = error - last_error
        D_term = Kd * derivative
        last_error = error
        
        # Tính toán giá trị điều khiển cuối cùng
        # Bắt đầu từ giá trị trung bình và cộng/trừ phần điều chỉnh của PID
        pid_adjustment = P_term + I_term + D_term
        output_throttle = neutral_throttle + pid_adjustment

        # --- GIỚI HẠN OUTPUT TRONG KHOẢNG [1000, 2000] ---
        output_throttle = max(1000, min(2000, output_throttle))
        
        print(f"Error: {error:4.0f} | PID: {int(pid_adjustment):4d} | Output: {int(output_throttle)}")
        
        # Ngủ một chút để không làm quá tải CPU
        time.sleep(0.02) # Tần suất khoảng 50Hz

    print("PID Controller: Đã thoát.")
    
# Thêm error_queue vào danh sách tham số
def object_detection_and_display(queue, error_queue, confidence_thresh): 
    """Tiến trình nhận diện vật thể, hiển thị, và gửi sai số."""
    # ... (phần code đầu hàm giữ nguyên) ...

    while True:
        frame = queue.get()
        if frame is None:
            # Gửi tín hiệu kết thúc cho tiến trình PID trước khi thoát
            error_queue.put(None) 
            break
        
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        bbox, label, conf = detect_common_objects(frame, model="yolov3", confidence=confidence_thresh)
        height, width, _ = frame.shape
        center_frame_x = width // 2
        im_with_detection = draw_bbox(frame, bbox, label, conf)
        
        person_detected = False
        for box, lbl, cf in zip(bbox, label, conf):
            if lbl == 'person':
                person_detected = True
                x1, y1, x2, y2 = box
                center_x_person = int((x1 + x2) / 2)
                center_y_person = int((y1 + y2) / 2)

                # --- TÍNH TOÁN SAI SỐ (ERROR) ---
                # Sai số là khoảng cách từ tâm người đến tâm khung hình theo trục X
                # Dấu (+): người ở bên trái, cần di chuyển sang phải
                # Dấu (-): người ở bên phải, cần di chuyển sang trái
                error = center_frame_x - center_x_person
                
                # --- GỬI SAI SỐ CHO TIẾN TRÌNH PID ---
                if error_queue.qsize() < 5:
                    error_queue.put(error)

                # ... (phần code vẽ vời giữ nguyên) ...
                cv2.circle(im_with_detection, (center_x_person, center_y_person), 5, (0, 0, 255), -1)
                cv2.line(im_with_detection, (center_x_person, center_y_person), (center_frame_x, center_y_person), (0, 255, 255), 2)
                cv2.putText(im_with_detection, f"Error: {error}", (center_x_person + 10, center_y_person - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        # Nếu không phát hiện thấy người, gửi sai số = 0
        if not person_detected:
            if error_queue.qsize() < 5:
                error_queue.put(0)

        cv2.imshow('Object Detection', im_with_detection)
        if cv2.waitKey(1) == ord('q'):
            # Gửi tín hiệu kết thúc cho tiến trình PID nếu thoát bằng 'q'
            error_queue.put(None)
            break

    cv2.destroyAllWindows()
    print("Object Detection: Đã thoát.")
if __name__ == '__main__':
    print("Ứng dụng nhận diện vật thể khởi động.")
    
    # --- CÁC HẰNG SỐ PID (CẦN HIỆU CHỈNH THỰC TẾ) ---
    # Kp: Phản ứng với sai số hiện tại. Kp lớn -> phản ứng mạnh.
    # Ki: Xử lý sai số tích lũy theo thời gian, giúp loại bỏ sai số ổn định.
    # Kd: Phản ứng với tốc độ thay đổi của sai số, giúp giảm vọt lố và ổn định hệ thống.
    Kp = 0.4
    Ki = 0.02
    Kd = 0.1
    # -----------------------------------------------------

    # Tạo các Queue để truyền dữ liệu giữa các tiến trình
    frame_queue = mp.Queue(maxsize=10) # Hàng đợi khung hình
    error_queue = mp.Queue(maxsize=5)  # Hàng đợi giá trị sai số

    # Tạo các tiến trình
    # Tiến trình 1: Đọc khung hình
    p_grabber = mp.Process(target=frame_grabber, args=(frame_queue, camera_index))
    
    # Tiến trình 2: Nhận diện, hiển thị và gửi sai số
    p_detector = mp.Process(target=object_detection_and_display, args=(frame_queue, error_queue, confidence_threshold))

    # Tiến trình 3: Tính toán PID
    p_pid = mp.Process(target=pid_controller, args=(error_queue, Kp, Ki, Kd))

    # Bắt đầu các tiến trình
    p_grabber.start()
    p_detector.start()
    p_pid.start()

    # Chờ các tiến trình kết thúc
    p_grabber.join()
    p_detector.join()
    p_pid.join()

    print("Tất cả các tiến trình đã kết thúc.")
    print("Ứng dụng đã thoát hoàn toàn.")