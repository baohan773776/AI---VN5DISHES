import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.utils import img_to_array
from tensorflow.keras.models import load_model

MODEL_PATH = r'D:\UEH\2 - HKĐ2026\AI\BUỔI 4\VIETNAMESE DISHES\model.keras'

print(f"Đang tải model {MODEL_PATH}...")
model = load_model(MODEL_PATH)

class_labels = {
    0: "BÁNH CUỐN",
    1: "BÁNH MÌ",
    2: "BÁNH XÈO",
    3: "BÚN ĐẬU MẮM TÔM",
    4: "BÁNH CANH"
}

class_labels_no_accent = {
    0: "Banh cuon",
    1: "Banh mi",
    2: "Banh xeo",
    3: "Bun dau mam tom",
    4: "Banh canh"
}

IMG_WIDTH, IMG_HEIGHT = 128, 128


class DishRecognitionGUI:
    def __init__(self, root, model, class_labels, class_labels_no_accent):
        self.root = root
        self.model = model
        self.class_labels = class_labels
        self.class_labels_no_accent = class_labels_no_accent
        self.cap = None
        self.current_frame = None
        self.is_webcam_running = False
        self.is_frozen = False

        self.root.title("Vietnamese Food Recognition AI System")
        self.root.geometry("1100x650")

        self.left_panel = tk.Label(root, bg="black")
        self.left_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.right_panel = tk.Frame(root, width=360, bg="#111111")
        self.right_panel.pack(side="right", fill="y", padx=10, pady=10)
        self.right_panel.pack_propagate(False)

        tk.Label(
            self.right_panel,
            text="KẾT QUẢ NHẬN DIỆN MÓN ĂN",
            font=("Segoe UI", 14, "bold"),
            fg="#00ffff",
            bg="#111111"
        ).pack(pady=(15, 10))

        self.result_text = tk.Text(
            self.right_panel,
            font=("Segoe UI", 12),
            fg="white",
            bg="#111111",
            wrap="word",
            borderwidth=0,
            height=22
        )
        self.result_text.pack(fill="both", expand=True, padx=15)

        tk.Button(
            self.right_panel,
            text="Mở webcam realtime",
            font=("Segoe UI", 11),
            command=self.start_webcam
        ).pack(fill="x", padx=15, pady=5)

        tk.Button(
            self.right_panel,
            text="Import ảnh món ăn",
            font=("Segoe UI", 11),
            command=self.import_image
        ).pack(fill="x", padx=15, pady=5)

        tk.Button(
            self.right_panel,
            text="Q - Thoát",
            font=("Segoe UI", 11),
            command=self.close
        ).pack(fill="x", padx=15, pady=(5, 15))

        self.root.bind("<c>", lambda event: self.capture_webcam_frame())
        self.root.bind("<C>", lambda event: self.capture_webcam_frame())
        self.root.bind("<r>", lambda event: self.reset_webcam())
        self.root.bind("<R>", lambda event: self.reset_webcam())
        self.root.bind("<q>", lambda event: self.close())
        self.root.bind("<Q>", lambda event: self.close())
        self.root.bind("<Escape>", lambda event: self.close())

        self.show_message([
            "Mô hình hỗ trợ 5 món ăn: Bánh cuốn, Bánh mì, Bánh xèo, Bún đậu mắm tôm, Bánh canh",
            "Chọn một chế độ điều khiển:",
            "- Mở webcam realtime để nhận diện món ăn trực tiếp.",
            "- Nhấn C trên bàn phím để chụp/chốt kết quả.",
            "- Nhấn R để khởi động lại camera.",
            "- Bấm 'Import ảnh món ăn' để kiểm tra từ tệp ảnh có sẵn."
        ])

    def show_message(self, lines):
        self.result_text.delete("1.0", tk.END)
        for line in lines:
            self.result_text.insert(tk.END, line + "\n\n")

    def predict_frame(self, frame):
        img_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype("float32") / 255.0

        pred_probs = self.model.predict(img_array, verbose=0)
        pred_idx = np.argmax(pred_probs, axis=1)[0]

        dish_name_with_accent = self.class_labels.get(pred_idx, "Không rõ")
        dish_name_no_accent = self.class_labels_no_accent.get(pred_idx, "Unknown")
        confidence = pred_probs[0][pred_idx] * 100

        annotated_frame = frame.copy()
        display_text = f"{dish_name_no_accent}: {confidence:.2f}%"

        cv2.rectangle(annotated_frame, (10, 15), (450, 65), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            display_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        reading_report = [
            f"🍲 PHÂN TÍCH THỰC PHẨM THÀNH CÔNG:",
            f"📌 Món ăn dự đoán: {dish_name_with_accent}",
            f"🎯 Độ chính xác: {confidence:.2f}%",
            f"⚡ Trạng thái: Bộ lọc hoạt động ổn định"
        ]

        return annotated_frame, reading_report

    def show_frame_on_gui(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((720, 600))
        imgtk = ImageTk.PhotoImage(image=img)
        self.left_panel.imgtk = imgtk
        self.left_panel.configure(image=imgtk)

    def start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_message(["Không thể kết nối và mở được thiết bị webcam."])
            return
        self.is_webcam_running = True
        self.is_frozen = False
        self.show_message([
            "🎥 CHẾ ĐỘ WEBCAM TRỰC TIẾP",
            "Đang quét luồng video realtime...",
            "Đưa đĩa/bát món ăn vào chính giữa khung hình.",
            "Nhấn C để chụp/chốt kết quả.",
            "Nhấn R để tải lại (Restart).",
            "Nhấn Q để tắt chương trình."
        ])
        self.update_webcam()

    def update_webcam(self):
        if not self.is_webcam_running or self.is_frozen:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.show_message(["Không đọc được dữ liệu frame từ webcam."])
            return
        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()
        annotated_frame, _ = self.predict_frame(frame)
        cv2.putText(
            annotated_frame,
            "LIVE - C: CAPTURE | R: RESTART | Q: QUIT",
            (20, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        self.show_frame_on_gui(annotated_frame)
        self.root.after(30, self.update_webcam)

    def capture_webcam_frame(self):
        if self.current_frame is None:
            self.show_message(["Không tìm thấy dữ liệu ảnh để chụp chốt."])
            return
        self.is_frozen = True
        self.is_webcam_running = False
        annotated_frame, reading = self.predict_frame(self.current_frame)
        cv2.putText(
            annotated_frame,
            "CAPTURED - R: RESTART | Q: QUIT",
            (20, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        self.show_frame_on_gui(annotated_frame)
        self.show_message(reading)

    def reset_webcam(self):
        self.is_frozen = False
        self.is_webcam_running = True
        self.show_message([
            "🔄 ĐÃ TẢI LẠI WEBCAM",
            "Đang tiếp tục quét tìm món ăn...",
            "Đảm bảo đĩa thức ăn nhận đủ ánh sáng tốt."
        ])
        self.update_webcam()

    def import_image(self):
        self.is_webcam_running = False
        self.is_frozen = True
        file_path = filedialog.askopenfilename(
            title="Chọn hình ảnh món ăn Việt Nam",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return
        try:
            with open(file_path, "rb") as f:
                chunk = f.read()
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            frame = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            frame = None
            print(f"Lỗi đọc file nhị phân: {e}")
        if frame is None:
            self.show_message(["Định dạng tập tin lỗi hoặc không đọc được hình ảnh."])
            return
        self.current_frame = frame.copy()
        annotated_frame, reading = self.predict_frame(frame)
        cv2.putText(
            annotated_frame,
            "IMAGE MODE - R: WEBCAM | Q: QUIT",
            (20, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )
        self.show_frame_on_gui(annotated_frame)
        self.show_message(reading)

    def close(self):
        self.is_webcam_running = False
        self.is_frozen = True
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()
        print("Đã giải phóng tài nguyên hệ thống và đóng cửa sổ GUI.")


if __name__ == "__main__":
    root = tk.Tk()
    app = DishRecognitionGUI(root, model, class_labels, class_labels_no_accent)
    root.mainloop()