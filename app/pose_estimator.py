import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
import os
import tempfile
from tqdm import tqdm  # 导入 tqdm 库用于显示进度条


# 初始化 MMPose Inferencer (选择一个模型)
MODEL_ALIAS = 'td-hm_ViTPose-base_8xb64-210e_coco-256x192'  # 或者 'hrnet_w48_coco_256x192'
INFERENCER = None


def get_inferencer():
    global INFERENCER
    if INFERENCER is None:
        print(f"Initializing MMPoseInferencer with model: {MODEL_ALIAS}")
        INFERENCER = MMPoseInferencer(pose2d=MODEL_ALIAS, device='cuda:0')
        print("MMPoseInferencer initialized.")
    return INFERENCER


def extract_poses_from_video(video_path: str, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    inferencer = get_inferencer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    all_keypoints = []
    frame_idx = 0

    # 使用 inferencer.infer 进行视频处理
    result_generator = inferencer(video_path, return_vis=False,
                                  out_dir=output_dir)  # out_dir 用于保存可视化结果（如果return_vis=True）
    results = [result for result in result_generator]

    # 添加进度条
    processed_frames = 0
    with tqdm(total=len(results), desc="Processing frames", unit="frame") as pbar:
        for frame_result in results:
            if 'predictions' in frame_result and len(frame_result['predictions']) > 0:
                person_prediction = frame_result['predictions'][0]
                if len(person_prediction) > 0:
                    keypoints_data = person_prediction[0]
                    keypoints = np.array(keypoints_data['keypoints']).tolist()
                    keypoint_scores = np.array(keypoints_data['keypoint_scores']).tolist()

                    combined_keypoints = []
                    for kp, score in zip(keypoints, keypoint_scores):
                        combined_keypoints.append(kp + [score])  # [x, y, score]

                    all_keypoints.append({
                        "frame_id": processed_frames,
                        "keypoints": combined_keypoints
                    })
                else:  # No person detected in this frame by the model
                    all_keypoints.append({
                        "frame_id": processed_frames,
                        "keypoints": []
                    })
            else:  # No predictions for this frame
                all_keypoints.append({
                    "frame_id": processed_frames,
                    "keypoints": []
                })

            processed_frames += 1
            pbar.update(1)  # 更新进度条

    cap.release()

    # 获取视频基本信息
    cap_temp = cv2.VideoCapture(video_path)
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_val = cap_temp.get(cv2.CAP_PROP_FPS)
    cap_temp.release()

    # 完成提示
    print("\nVideo processing completed successfully!")

    return {
        "video_info": {"width": width, "height": height, "fps": fps_val, "total_frames": processed_frames},
        "pose_data": all_keypoints
    }


def extract_pose_from_image(image_path: str):
    inferencer = get_inferencer()
    result_generator = inferencer(image_path, return_vis=False)
    results = [result for result in result_generator]

    if results and 'predictions' in results[0] and len(results[0]['predictions']) > 0:
        person_prediction = results[0]['predictions'][0]
        if len(person_prediction) > 0:
            keypoints_data = person_prediction[0]
            keypoints = np.array(keypoints_data['keypoints']).tolist()
            keypoint_scores = np.array(keypoints_data['keypoint_scores']).tolist()
            combined_keypoints = [kp + [score] for kp, score in zip(keypoints, keypoint_scores)]

            img = cv2.imread(image_path)
            height, width, _ = img.shape

            # 完成提示
            print("\nImage processing completed successfully!")

            return {
                "image_info": {"width": width, "height": height},
                "pose_data": [{"frame_id": 0, "keypoints": combined_keypoints}]  # Treat as a single frame video
            }
    return {"image_info": {}, "pose_data": []}