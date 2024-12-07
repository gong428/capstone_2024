import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 동작 이름
actions = ['점수 작성']
seq_length = 90

# 데이터셋 폴더 경로
data_path = 'dataset'  # 데이터셋이 저장된 폴더

# 데이터 전처리: 포즈 데이터 추출
def extract_pose_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    data = []

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img)

        if result.pose_landmarks is not None:
            joint = np.zeros((33, 4))
            for j, lm in enumerate(result.pose_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # 부모-자식 조인트 간 벡터 및 각도 계산
            v1 = joint[[0, 1, 2, 3, 0, 4, 5, 6, 10, 20, 18, 20, 16, 16, 14, 12, 12, 11, 11, 13, 15, 15, 15, 17, 24, 24, 26, 28, 28, 30, 23, 25, 27, 27, 31], :3]
            v2 = joint[[1, 2, 3, 7, 4, 5, 6, 8, 9, 18, 16, 16, 22, 14, 12, 11, 24, 13, 23, 15, 21, 17, 19, 19, 23, 26, 28, 30, 32, 32, 25, 27, 29, 31, 29], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n', v[::], v[1::]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            data.append(d)

    cap.release()
    return np.array(data)

# 데이터셋 준비
def prepare_dataset():
    sequences, labels = [], []
    for idx, action in enumerate(actions):
        action_folder = os.path.join(data_path, action)
        for video in os.listdir(action_folder):
            video_path = os.path.join(action_folder, video)
            pose_data = extract_pose_landmarks(video_path)

            for start in range(0, len(pose_data) - seq_length + 1):
                sequences.append(pose_data[start:start + seq_length])
                labels.append(idx)

    return np.array(sequences), np.array(labels)

# 데이터 준비
X_data, y_data = prepare_dataset()
y_data = np.eye(len(actions))[y_data]  # One-hot encoding

# 데이터셋 분할
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(seq_length, X_data.shape[2])),
    LSTM(128, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 체크포인트 설정
checkpoint = ModelCheckpoint('models/model_ver_4.6.h5', monitor='val_loss', save_best_only=True, verbose=1)

# 모델 학습
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[checkpoint])

print("모델 학습 완료")
