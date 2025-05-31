import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from datetime import datetime
import json
import os
from tkinter import messagebox
import openpyxl

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from keras import layers

behaviors_mapping = {
    1: "Vỗ tay",
    2: "Lắc lư",
    3: "Quay tròn",
    4: "Gõ ngón tay",
    5: "Chuyển động ngón tay",
    6: "Xếp đồ vật",
    7: "Quay đồ vật",
    8: "Mở và đóng cửa",
    9: "Di chuyển đồ vật",
    10: "Phân loại đồ vật"
}

@tf.keras.utils.register_keras_serializable(package="ViViT")
class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.projection.filters,
            "patch_size": self.projection.kernel_size
        })
        return config

@tf.keras.utils.register_keras_serializable(package="ViViT")
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(0, num_tokens, 1)

    def call(self, encoded_tokens):
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config

# Load model ViViT
custom_objects = {"TubeletEmbedding": TubeletEmbedding, "PositionalEncoder": PositionalEncoder}
model = load_model("vivit_model.keras", custom_objects=custom_objects)


def save_history():
    history_file = "history.json"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "video": video_path.get(),
        "result": result_var.get(),
        "level": autism_level.get(),
        "time": now
    }

    history_data = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except json.JSONDecodeError:
            history_data = []

    history_data.append(entry)

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=4)

# Hàm xử lý sự kiện
def upload_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if filepath:
        video_path.set(filepath)
    file_name, codec, size, resolution, duration, fps, frame_count = get_video_info(filepath)

    # Cập nhật nội dung của video_label để hiển thị thông tin
    video_label.config(
        text=f"Tên: {file_name}\nĐịnh dạng: {codec}\nDung lượng: {size}\nĐộ phân giải: {resolution}\nThời gian: {duration}\nFPS: {fps}\nSố frame: {frame_count}")

def get_video_info(filepath):
    """Lấy thông tin video gồm tên file, định dạng, dung lượng, thời gian phát, tỷ lệ khung hình."""
    if not os.path.exists(filepath):
        return "File không tồn tại!", "", "", "", "", "", ""

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return "Không thể mở video!", "", "", "", "", "", ""

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Chiều rộng
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Chiều cao
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # Định dạng codec
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Tổng số frame
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Tỷ lệ khung hình
    duration = frame_count / fps if fps > 0 else 0  # Tính thời gian phát

    cap.release()

    codec = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])  # Giải mã codec
    file_name = os.path.basename(filepath)  # Tên file
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Dung lượng file (MB)

    return file_name, codec, f"{file_size:.2f} MB", f"{width} x {height}", f"{duration:.2f} giây", f"{fps} FPS", frame_count

def process_video():
    threshold = 0.4
    """Xử lý video, chia thành các đoạn 5 giây, chuẩn hóa kích thước, và dự đoán hành vi."""

    filepath = video_path.get()  # Lấy đường dẫn video từ Tkinter StringVar
    if not filepath or not os.path.exists(filepath):
        print("Lỗi: Video không tồn tại hoặc đường dẫn không hợp lệ!")
        return

    cap = cv2.VideoCapture(filepath)  # Mở video
    if not cap.isOpened():
        print("Lỗi: Không thể mở video!")
        return

    cap = cv2.VideoCapture(filepath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Lấy số frame/giây
    frames_per_segment = min(fps * 5, 32)  # Giới hạn tối đa 32 frame mỗi lần dự đoán

    frames = []
    predictions = []
    time_stamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_stamp += 1 / fps  # Cập nhật thời gian theo từng frame
        frame_resized = cv2.resize(frame, (56, 56))
        frames.append(frame_resized)

        if len(frames) == frames_per_segment:
            # Đảm bảo kích thước đúng với yêu cầu mô hình
            frames_array = np.expand_dims(np.array(frames), axis=0)  # (1, 32, 56, 56, 3)

            # Dự đoán hành vi từ đoạn video 5 giây
            prediction_scores = model.predict(frames_array)[0]
            max_score = np.max(prediction_scores)
            predicted_behavior = np.argmax(prediction_scores) + 1

            # Kiểm tra ngưỡng tin cậy
            if max_score < threshold:
                predictions.append((time_stamp/60, "Không xác định"))
            else:
                predictions.append((time_stamp/60, behaviors_mapping[predicted_behavior]))
                behavior_name = behaviors_mapping.get(predicted_behavior, "Không xác định")
                print(f"{time_stamp/60:.2f}s - {behavior_name}")
                update_behavior_list(f"{time_stamp / 60:.2f} - {behavior_name}")

            frames = []  # Reset frame để xử lý đoạn tiếp theo

    cap.release()

    # Xử lý phần dư nếu video kết thúc nhưng chưa đủ 32 frame
    if frames:
        while len(frames) < 32:
            blank_frame = np.ones((56, 56, 3), dtype=np.uint8) * 255
            frames.append(blank_frame)
        frames_array = np.expand_dims(np.array(frames), axis=0)

        prediction_scores = model.predict(frames_array)[0]
        max_score = np.max(prediction_scores)
        predicted_behavior = np.argmax(prediction_scores) + 1

        if max_score < threshold:
            predictions.append((time_stamp/60, "Không xác định"))
        else:
            predictions.append((time_stamp/60, behaviors_mapping[predicted_behavior]))
            behavior_name = behaviors_mapping.get(predicted_behavior, "Không xác định")
            print(f"{time_stamp/60:.2f}s - {behavior_name}")
            update_behavior_list(f"{time_stamp/60:.2f} - {behavior_name}")

    return predictions

def start_analysis():
    process_video()
    result_var.set("75%")
    autism_level.set("CAO")
    save_history()

def update_behavior_list(behavior_info):
    behavior_list.insert(tk.END, behavior_info)

def show_history():
    history_win = tk.Toplevel(root)
    history_win.title("Lịch sử sàng lọc")
    history_win.geometry("700x400")

    columns = ("video", "result", "level", "time")

    tree = ttk.Treeview(history_win, columns=columns, show="headings")
    tree.heading("video", text="Video")
    tree.heading("result", text="Kết quả (%)")
    tree.heading("level", text="Mức độ")
    tree.heading("time", text="Thời gian")

    tree.column("video", width=300)
    tree.column("result", width=80)
    tree.column("level", width=80)
    tree.column("time", width=180)

    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Load dữ liệu từ file JSON
    history_file = "history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    tree.insert("", tk.END, values=(item["video"], item["result"], item["level"], item["time"]))
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi đọc history.json:\n{e}")

    # Xóa lịch sử
    def clear_history():
        if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa toàn bộ lịch sử?"):
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump([], f)
            for item in tree.get_children():
                tree.delete(item)
            messagebox.showinfo("Đã xóa", "Toàn bộ lịch sử đã được xóa.")

    # Xuất file Excel
    def export_excel():
        if not os.path.exists(history_file):
            messagebox.showerror("Lỗi", "Không tìm thấy file lịch sử.")
            return

        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            messagebox.showwarning("Trống", "Không có dữ liệu để xuất.")
            return

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Lịch sử"

        # Tiêu đề
        ws.append(["Video", "Kết quả (%)", "Mức độ", "Thời gian"])

        # Dữ liệu
        for item in data:
            ws.append([item["video"], item["result"], item["level"], item["time"]])

        # Lưu
        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if save_path:
            wb.save(save_path)
            messagebox.showinfo("Thành công", f"Đã lưu file Excel tại:\n{save_path}")

    # Nút chức năng
    btn_frame = tk.Frame(history_win)
    btn_frame.pack(pady=5)

    tk.Button(btn_frame, text="Xóa lịch sử", bg="red", fg="white", command=clear_history).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Xuất Excel", bg="green", fg="white", command=export_excel).pack(side=tk.LEFT, padx=10)

def show_help():
    help_win = tk.Toplevel(root)
    help_win.title("Hướng dẫn sử dụng")
    help_text = """
    1. Tải video quay trẻ khi chơi, giao tiếp.
    2. Nhấn 'Bắt đầu phân tích'.
    3. Xem danh sách hành vi lặp lại.
    4. Đọc kết quả mức độ tự kỷ.
    """
    tk.Label(help_win, text=help_text, justify=tk.LEFT).pack(padx=10, pady=10)

def show_behavior_settings():
    settings_win = tk.Toplevel(root)
    settings_win.title("Thiết lập mức độ tự kỷ theo hành vi")
    settings_win.geometry("700x400")

    headers = ["STT", "Hành vi", "Số giây xuất hiện tối thiểu", "Hành vi lặp lại (lần)", "Mức độ tự kỷ (%)"]
    behaviors = [
        "Vỗ tay", "Lắc lư", "Quay tròn", "Gõ ngón tay, gõ chân",
        "Chuyển động ngón tay, chân", "Xếp đồ vật theo hàng, thứ tự",
        "Quay đồ vật", "Mở và đóng cửa", "Di chuyển đồ vật lặp lại", "Phân loại đồ vật"
    ]

    # Danh sách lưu các Entry để xử lý
    entries = []

    # Load dữ liệu cũ nếu tồn tại
    loaded_data = {}
    if os.path.exists("behavior_settings.json"):
        try:
            with open("behavior_settings.json", "r", encoding="utf-8") as f:
                json_data = json.load(f)
                for item in json_data:
                    loaded_data[item["hanh_vi"]] = {
                        "thoi_gian": item["thoi_gian"],
                        "lap_lai": item["lap_lai"],
                        "muc_do": item["muc_do"]
                    }
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi đọc file JSON:\n{e}")

    # Tiêu đề bảng
    for col, h in enumerate(headers):
        width = 6 if col == 0 else 20
        tk.Label(settings_win, text=h, borderwidth=1, relief="solid", width=width, bg="lightgray").grid(row=0, column=col)

    # Các dòng dữ liệu
    for i, behavior in enumerate(behaviors):
        tk.Label(settings_win, text=str(i+1), borderwidth=0, relief="solid", width=6).grid(row=i+1, column=0)
        tk.Label(settings_win, text=behavior, borderwidth=0, relief="solid", width=20, anchor="w").grid(row=i+1, column=1)

        row_entries = []
        for j, key in enumerate(["thoi_gian", "lap_lai", "muc_do"]):
            entry = tk.Entry(settings_win, width=10)

            # Nếu có dữ liệu cũ, gán vào ô nhập
            if behavior in loaded_data:
                entry.insert(0, str(loaded_data[behavior][key]))

            entry.grid(row=i+1, column=j+2)
            row_entries.append(entry)

        entries.append(row_entries)

    # Hàm lưu lại thiết lập
    def save_settings():
        data = []
        for i, behavior in enumerate(behaviors):
            try:
                duration = float(entries[i][0].get())
                repeat = int(entries[i][1].get())
                percentage = float(entries[i][2].get())
            except ValueError:
                messagebox.showerror("Lỗi", f"Dữ liệu không hợp lệ ở dòng {i+1}")
                return

            data.append({
                "ten_nhan": i + 1,
                "hanh_vi": behavior,
                "thoi_gian": duration,
                "lap_lai": repeat,
                "muc_do": percentage
            })

        with open("behavior_settings.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("Thành công", "Đã lưu thiết lập vào behavior_settings.json")

    # Nút lưu
    tk.Button(settings_win, text="Lưu thiết lập", command=save_settings, bg="blue", fg="white").grid(row=len(behaviors) + 1, column=0, columnspan=5, pady=10)

def show_autism_level_settings():
    level_win = tk.Toplevel(root)
    level_win.title("Thiết lập kết qủa mức độ tự kỷ")
    level_win.geometry("400x250")

    levels = ["Thấp", "Trung bình", "Cao", "Rất cao"]
    entries = {}

    # Load dữ liệu cũ nếu có
    loaded_levels = {}
    if os.path.exists("autism_levels.json"):
        try:
            with open("autism_levels.json", "r", encoding="utf-8") as f:
                loaded_levels = json.load(f)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi đọc autism_levels.json:\n{e}")

    # Tiêu đề
    tk.Label(level_win, text="Tên mức độ", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=10, pady=5)
    tk.Label(level_win, text="Từ (%)", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=10)
    tk.Label(level_win, text="Đến (%)", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=10)

    # Tạo dòng nhập cho mỗi mức
    for i, level in enumerate(levels):
        tk.Label(level_win, text=level).grid(row=i+1, column=0, sticky="w", padx=10)

        entry_from = tk.Entry(level_win, width=8)
        entry_to = tk.Entry(level_win, width=8)

        # Nạp dữ liệu cũ nếu có
        if level in loaded_levels:
            entry_from.insert(0, str(loaded_levels[level]["from"]))
            entry_to.insert(0, str(loaded_levels[level]["to"]))

        entry_from.grid(row=i+1, column=1, padx=5)
        entry_to.grid(row=i+1, column=2, padx=5)

        entries[level] = (entry_from, entry_to)

    # Hàm lưu thiết lập
    def save_levels():
        data = {}
        for level in levels:
            try:
                from_val = float(entries[level][0].get())
                to_val = float(entries[level][1].get())
                if from_val > to_val:
                    messagebox.showerror("Lỗi", f"{level}: 'Từ' phải nhỏ hơn hoặc bằng 'Đến'")
                    return
                data[level] = {"from": from_val, "to": to_val}
            except ValueError:
                messagebox.showerror("Lỗi", f"{level}: Giá trị không hợp lệ")
                return

        with open("autism_levels.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("Thành công", "Đã lưu thiết lập vào autism_levels.json")

    # Nút lưu
    tk.Button(level_win, text="Lưu thiết lập", command=save_levels, bg="blue", fg="white").grid(row=6, column=0, columnspan=3, pady=20)

# Giao diện chính
root = tk.Tk()
root.title("PHẦN MỀM HỖ TRỢ SÀNG LỌC TRẺ TỰ KỶ")
root.geometry("700x400")
root.resizable(True, True)

# Thêm menu vào đầu chương trình
menubar = tk.Menu(root)
history_menu = tk.Menu(menubar, tearoff=0)
history_menu.add_command(label="Lịch sử sàng lọc", command=show_history)
menubar.add_cascade(label="Lịch sử", menu=history_menu)

settings_menu = tk.Menu(menubar, tearoff=0)
settings_menu.add_command(label="Thiết lập hành vi", command=show_behavior_settings)
settings_menu.add_command(label="Thiết lập mức độ tự kỷ", command=show_autism_level_settings)
menubar.add_cascade(label="Cấu hình", menu=settings_menu)

root.config(menu=menubar)


# Biến
video_path = tk.StringVar()
result_var = tk.StringVar(value="0%")
autism_level = tk.StringVar(value="")

# Khung chọn video
top_frame = tk.Frame(root, padx=10, pady=10)
top_frame.pack(fill=tk.X)

entry = tk.Entry(top_frame, textvariable=video_path, width=50)
entry.pack(side=tk.LEFT, padx=5)

btn_upload = tk.Button(top_frame, text="TẢI LÊN VIDEO", command=upload_video)
btn_upload.pack(side=tk.LEFT, padx=5)

btn_start = tk.Button(top_frame, text="BẮT ĐẦU PHÂN TÍCH", bg="blue", fg="white", command=start_analysis)
btn_start.pack(side=tk.LEFT, padx=5)

# Khung hiển thị
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Video Play placeholder
filepath = video_path.get()
file_name, codec, size, resolution, duration, fps, frame_count = get_video_info(filepath)
video_frame = tk.LabelFrame(main_frame, text="THÔNG TIN VIDEO", width=150, height=1000)
video_frame.pack(side=tk.LEFT, padx=5)
video_label = tk.Label(video_frame, text="Chưa có video", bg="#e0e0e0", width=40, height=30)
video_label.pack()


# Danh sách hành vi
behavior_frame = tk.LabelFrame(main_frame, text="HÀNH VI LẶP LẠI", width=200)
behavior_frame.pack(side=tk.LEFT, padx=0, fill=tk.Y)

behavior_list = tk.Listbox(behavior_frame, height=30, width=30)
behavior_list.pack(padx=5, pady=5)

# Kết quả
result_frame = tk.LabelFrame(main_frame, text="MỨC ĐỘ TỰ KỶ", width=200)
result_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

style = ttk.Style()
style.configure("blue.Horizontal.TProgressbar", foreground='blue', background='blue')

progress = ttk.Progressbar(result_frame, style="blue.Horizontal.TProgressbar", orient="horizontal", length=150, mode="determinate")
progress["value"] = 75
progress.pack(pady=20, padx=20)

result_label = tk.Label(result_frame, textvariable=result_var, font=("Arial", 16))
result_label.pack()

level_label = tk.Label(result_frame, textvariable=autism_level, font=("Arial", 12), bg="red", fg="white", width=10)
level_label.pack()

# Main loop
root.mainloop()
