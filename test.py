import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
def get_grade(video_path):
    last_5_predictions = []
    actions = ['점수 작성']
    seq_length = 90
    max_continuous_length = 5

    model = load_model('models/model_ver_4.6.h5')

    prev_action = None
    continuous_length = 0

    # MediaPipe pose model
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    data = []

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        landmarks = result.pose_landmarks

        if landmarks is not None:
            joint = np.zeros((33, 4))
            for j, lm in enumerate(landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 4, 5, 6, 10, 20, 18, 20, 16, 16, 14, 12, 12, 11, 11, 13, 15, 15, 15, 17, 24, 24, 26, 28, 28, 30, 23, 25, 27, 27, 31], :3]  # Parent joint
            v2 = joint[[1, 2, 3, 7, 4, 5, 6, 8, 9, 18, 16, 16, 22, 14, 12, 11, 24, 13, 23, 15, 21, 17, 19, 19, 23, 26, 28, 30, 32, 32, 25, 27, 29, 31, 29], :3]  # Child joint
            v = v2 - v1  # [20, 3]
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 3, 0, 4, 5, 6, 10, 20, 18, 20, 16, 16, 14, 12, 12, 11, 11, 13, 15, 15, 15, 17, 24, 24, 26, 28, 28, 30, 23, 25, 27, 27, 31], :],
                                        v[[1, 2, 3, 7, 4, 5, 6, 8, 9, 18, 16, 16, 22, 14, 12, 11, 24, 13, 23, 15, 21, 17, 19, 19, 23, 26, 28, 30, 32, 32, 25, 27, 29, 31, 29], :]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            data.append(d)

            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if len(data) >= seq_length:
                x_data = np.array(data[-seq_length:]).reshape(1, seq_length, -1)
                y_pred = model.predict(x_data)[0]
                pred_index = np.argmax(y_pred)
                pred_action = actions[pred_index]
                if pred_action == prev_action:
                    continuous_length += 1
                else:
                    # 현재 동작이 이전 동작과 다른 경우
                    continuous_length = 1
                    prev_action = pred_action

                # 가장 긴 연속 길이가 max_continuous_length보다 크거나 같으면 최종 등급으로 결정
                if continuous_length >= max_continuous_length:
                    final_action = pred_action
                    
                print(f"Predicted action: {pred_action}")

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return final_action