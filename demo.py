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

video_path = "D:/Projects/autism/video_dataset/15.mp4"

def process_video(video_path, threshold=0.6):
    """Xử lý video, chia thành các đoạn 5 giây, chuẩn hóa kích thước, và dự đoán hành vi."""
    cap = cv2.VideoCapture(video_path)
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
                #print(f"{time_stamp:.2f}s - Không xác định")
            else:
                predictions.append((time_stamp/60, behaviors_mapping[predicted_behavior]))
                behavior_name = behaviors_mapping.get(predicted_behavior, "Không xác định")
                print(f"{time_stamp/60:.2f}s - {behavior_name}")

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
            #print(f"{time_stamp:.2f}s - Không xác định")
        else:
            predictions.append((time_stamp/60, behaviors_mapping[predicted_behavior]))
            behavior_name = behaviors_mapping.get(predicted_behavior, "Không xác định")
            print(f"{time_stamp/60:.2f}s - {behavior_name}")

    return predictions


process_video(video_path)

# print("Chuỗi hành vi được nhận diện:")
# for time_stamp, behavior_id in predicted_behaviors:
#     behavior_name = behaviors_mapping.get(behavior_id, "Không xác định")
#     print(f"{time_stamp:.2f}s - {behavior_name}")