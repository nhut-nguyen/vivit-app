import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

# Hàm xử lý sự kiện
def upload_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if filepath:
        video_path.set(filepath)

def start_analysis():
    result_var.set("75%")
    autism_level.set("CAO")
    update_behavior_list()

def update_behavior_list():
    behaviors = [
        "00:12 - Vỗ tay",
        "00:14 - Lắc lư",
        "00:21 - Quay tròn",
        "00:35 - Vỗ tay",
        "00:38 - Quay tròn"
    ]
    for behavior in behaviors:
        behavior_list.insert(tk.END, behavior)

# Giao diện chính
root = tk.Tk()
root.title("PHẦN MỀM HỖ TRỢ SÀNG LỌC TRẺ TỰ KỶ")
root.geometry("700x400")
root.resizable(True, True)

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
video_frame = tk.LabelFrame(main_frame, text="", width=300, height=1000)
video_frame.pack(side=tk.LEFT, padx=5)
video_label = tk.Label(video_frame, text="VIDEO", bg="#e0e0e0", width=40, height=30)
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
