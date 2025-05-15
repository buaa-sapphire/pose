import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
import os
import tempfile

# 初始化 MMPose Inferencer (选择一个模型)
# 模型列表: https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html
# 例如，使用 HRNet: 'hrnet_w48_coco_256x192'
# 或 ViTPose: 'vitpose-base-coco-256x192'
# inferencer = MMPoseInferencer(pose2d='hrnet_w48_coco_256x192', device='cuda:0')
# 为了方便演示，这里先不实例化，在API调用时实例化或全局实例化
# 确保你的4090显卡被正确识别为 cuda:0

MODEL_ALIAS = 'td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314'  # 或者 'hrnet_w48_coco_256x192'
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
    # MMPoseInferencer可以直接处理视频路径，更高效
    result_generator = inferencer(video_path, return_vis=False,
                                  out_dir=output_dir)  # out_dir 用于保存可视化结果（如果return_vis=True）
    results = [result for result in result_generator]

    # 提取关键点
    # results[0]['predictions'][0][0]['keypoints'] # 这是一个例子，结构取决于模型
    # 确保从 results 中正确提取每帧的关键点

    # 为了简化，我们假设results的结构是这样的
    # result_generator 会生成一个迭代器，每个元素对应一帧的结果
    # 每个结果是一个dict, 'predictions' -> list of instances -> list of poses per instance
    # 我们假设每帧只有一个人，取第一个instance的第一个pose

    processed_frames = 0
    for frame_result in results:
        if 'predictions' in frame_result and len(frame_result['predictions']) > 0:
            # frame_result['predictions'] is a list of dicts, one for each detected person
            # Each dict has 'keypoints' (list of [x, y]), 'keypoint_scores' (list of scores)
            # Taking the first detected person's keypoints
            person_prediction = frame_result['predictions'][0]  # Assuming one list of predictions per frame
            if len(person_prediction) > 0:
                keypoints_data = person_prediction[0]  # Assuming one person
                keypoints = np.array(keypoints_data['keypoints']).tolist()
                keypoint_scores = np.array(keypoints_data['keypoint_scores']).tolist()

                # Combine keypoints with scores for simplicity in this example
                # In a real app, you might want to keep them separate or use scores for filtering
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

    cap.release()

    # 获取视频基本信息
    cap_temp = cv2.VideoCapture(video_path)
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_val = cap_temp.get(cv2.CAP_PROP_FPS)
    cap_temp.release()

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

            return {
                "image_info": {"width": width, "height": height},
                "pose_data": [{"frame_id": 0, "keypoints": combined_keypoints}]  # Treat as a single frame video
            }
    return {"image_info": {}, "pose_data": []}