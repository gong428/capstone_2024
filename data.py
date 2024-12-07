import cv2
import mediapipe as mp
import numpy as np
import time, os
def anal():
        print(os.path.abspath(__file__))
        #파트별로 이름 스윙 상체회전 
        actions = ['user']
        seq_length = 90
        secs_for_action = 300

        # MediaPipe pose model
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        #동영상 파일 길이 짧으면 오류 발생
        #0 웹캠 경로
        # 녹화한 영상 경로
        video_path = r'C:\Users\USER\Desktop\grade_video\video_d2.mp4'
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, 30)


        created_time = int(time.time())
        os.makedirs('dataset', exist_ok=True)

        while (cap.isOpened()):
            for idx, action in enumerate(actions):
                data = []

                ret, img = cap.read()

                cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.imshow('img', img)
                cv2.waitKey(1000)

                start_time = time.time()

                while time.time() - start_time < secs_for_action:
                    ret, img = cap.read()
                    if not ret:
                        break


                    img=cv2.flip(img,1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = pose.process(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                    landmarks= result.pose_landmarks
                    
                    

                    if result.pose_landmarks is not None:
                            joint = np.zeros((33, 4)) 
                            for  j,lm in enumerate(landmarks.landmark):
                                joint[j]= [lm.x, lm.y, lm.z, lm.visibility]
                            # Compute angles between joints
                            v1 = joint[[0,1,2,3,0,4,5,6,10,20,18,20,16,16,14,12,12,11,11,13,15,15,15,17,24,24,26,28,28,30,23,25,27,27,31], :3] # Parent joint
                            v2 = joint[[1,2,3,7,4,5,6,8,9,18,16,16,22,14,12,11,24,13,23,15,21,17,19,19,23,26,28,30,32,32,25,27,29,31,29], :3] # Child joint
                            v = v2 - v1 # [20, 3]
                            # Normalize v
                            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                            # Get angle using arcos of dot product
                            angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0,1,2,3,0,4,5,6,10,20,18,20,16,16,14,12,12,11,11,13,15,15,15,17,24,24,26,28,28,30,23,25,27,27,31],:], 
                                v[[1,2,3,7,4,5,6,8,9,18,16,16,22,14,12,11,24,13,23,15,21,17,19,19,23,26,28,30,32,32,25,27,29,31,29],:])) # [15,]

                            angle = np.degrees(angle) # Convert radian to degree

                            angle_label = np.array([angle], dtype=np.float32)
                            angle_label = np.append(angle_label, idx)

                            d = np.concatenate([joint.flatten(), angle_label])

                            data.append(d)

                            mp_drawing.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)

                    cv2.imshow('img', img)
                    if cv2.waitKey(1) == ord('q'):
                        break
                

                data = np.array(data)
                print(action, data.shape)
                np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

                # Create sequence data
                full_seq_data = []
                for seq in range(len(data) - seq_length):
                    full_seq_data.append(data[seq:seq + seq_length])

                full_seq_data = np.array(full_seq_data)
                print(action, full_seq_data.shape)
                np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
            break