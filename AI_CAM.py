import cv2
import cvlib as cv
import numpy as np
from cvlib.object_detection import detect_common_objects, draw_bbox
import time
import multiprocessing as mp # Import multiprocessing

# --- CẤU HÌNH ---
camera_index = 0 # Webcam mặc định của laptop
confidence_threshold = 0.7 # Ngưỡng tin cậy để phát hiện vật thể
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

def object_detection_and_display(queue, confidence_thresh):
    """Tiến trình nhận diện vật thể và hiển thị."""
    cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
    print("Object Detection: Bắt đầu xử lý...")

    while True:
        # Lấy khung hình từ hàng đợi
        frame = queue.get()
        if frame is None: # Nhận tín hiệu kết thúc từ Frame Grabber
            break

        # Thực hiện nhận diện vật thể
        bbox, label, conf = detect_common_objects(frame, model="yolov3-tiny", confidence=confidence_thresh)

        # Lấy kích thước của khung hình (chiều cao, chiều rộng, kênh)
        height, width, _ = frame.shape
        # Tính toán tâm của khung hình
        center_frame_x = width // 2
        center_frame_y = height // 2 # Chúng ta sẽ sử dụng điểm này để vẽ chấm giữa khung hình

        # Vẽ các hộp giới hạn và nhãn
        im_with_detection = draw_bbox(frame, bbox, label, conf)

        # --- VẼ ĐIỂM ĐỎ CHÍNH GIỮA NHÃN 'PERSON' & TÍNH KHOẢNG CÁCH ---
        for box, lbl, cf in zip(bbox, label, conf):
            if lbl == 'person':
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                # Tính toán tâm của hộp giới hạn "person"
                center_x_person = int((x1 + x2) / 2)
                center_y_person = int((y1 + y2) / 2)
                
                # Vẽ điểm tròn màu đỏ tại tâm của "person"
                cv2.circle(im_with_detection, (center_x_person, center_y_person), 5, (0, 0, 255), -1) # Màu đỏ (BGR)

                # Vẽ điểm xanh dương tại giữa khung hình theo trục X, nhưng cùng trục Y với person
                # (Điểm này là (center_frame_x, center_y_person) như bạn đã code trước đó)
                # Đây là điểm thứ hai để tính khoảng cách
                point_on_frame_x = center_frame_x
                point_on_frame_y = center_y_person # Cùng trục Y với center_y_person
                cv2.circle(im_with_detection, (point_on_frame_x, point_on_frame_y), 5, (255, 0, 0), -1) # Màu xanh dương (BGR)

                # --- TÍNH KHOẢNG CÁCH GIỮA HAI ĐIỂM ---
                # Điểm 1: Tâm của person (center_x_person, center_y_person)
                # Điểm 2: Điểm xanh dương trên trục giữa của khung hình (point_on_frame_x, point_on_frame_y)
                distance = np.sqrt(
                    (point_on_frame_x - center_x_person)**2 + 
                    (point_on_frame_y - center_y_person)**2
                )
                
                # In khoảng cách ra console
                print(f"Khoảng cách đến tâm màn hình (theo trục Y của person): {int(distance)} pixels")

                # (Tùy chọn) Vẽ đường thẳng nối hai điểm và hiển thị khoảng cách trên ảnh
                cv2.line(im_with_detection, 
                         (center_x_person, center_y_person), 
                         (point_on_frame_x, point_on_frame_y), 
                         (0, 255, 255), 2) # Màu vàng (BGR)

                cv2.putText(im_with_detection, 
                            f"Dist: {distance:.0f}", 
                            (center_x_person + 10, center_y_person - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- VẼ ĐIỂM ĐỎ CHÍNH GIỮA KHUNG HÌNH (Nếu bạn muốn một chấm cố định ở giữa ảnh) ---
        # Điểm này là (center_frame_x, center_frame_y)
        # Nếu bạn chỉ muốn chấm ở vị trí người, thì không cần dòng này
        # cv2.circle(im_with_detection, (center_frame_x, center_frame_y), 5, (0, 255, 255), -1) # Màu vàng (cho chấm chính giữa ảnh)

        cv2.imshow('Object Detection', im_with_detection)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Object Detection: Đã thoát.")

if __name__ == '__main__':
    print("Ứng dụng nhận diện vật thể khởi động.")
    
    # Tạo một Queue để truyền khung hình giữa các tiến trình
    frame_queue = mp.Queue()

    # Tạo các tiến trình
    # Tiến trình 1: Đọc khung hình
    p_grabber = mp.Process(target=frame_grabber, args=(frame_queue, camera_index))
    # Tiến trình 2: Nhận diện và hiển thị
    p_detector = mp.Process(target=object_detection_and_display, args=(frame_queue, confidence_threshold))

    # Bắt đầu các tiến trình
    p_grabber.start()
    p_detector.start()

    # Chờ các tiến trình kết thúc
    p_grabber.join()
    p_detector.join()

    print("Tất cả các tiến trình đã kết thúc.")
    print("Ứng dụng đã thoát hoàn toàn.")