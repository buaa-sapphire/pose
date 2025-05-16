import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
import os
import tempfile

import mmpose
import mmdet
import mmcv
print(f"MMPose: {mmpose.__version__}")
print(f"MMDetection: {mmdet.__version__}")
print(f"MMCV: {mmcv.__version__}")


# 初始化 MMPose Inferencer (选择一个模型)
# 模型列表: https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html
# 例如，使用 HRNet: 'hrnet_w48_coco_256x192'
# 或 ViTPose: 'vitpose-base-coco-256x192'
# inferencer = MMPoseInferencer(pose2d='hrnet_w48_coco_256x192', device='cuda:0')
# 为了方便演示，这里先不实例化，在API调用时实例化或全局实例化
# 确保你的4090显卡被正确识别为 cuda:0zzzenm

# MODEL_ALIAS = 'td-hm_ViTPose-base_8xb64-210e_coco-256x192'  # 或者 'hrnet_w48_coco_256x192'
MODEL_ALIAS = 'td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192'  # 或者 'hrnet_w48_coco_256x192'

# INFERENCER = None
#
#
# def get_inferencer():
#     global INFERENCER
#     if INFERENCER is None:
#         print(f"Initializing MMPoseInferencer with model: {MODEL_ALIAS}")
#         INFERENCER = MMPoseInferencer(pose2d=MODEL_ALIAS, device='cuda:0')
#         print("MMPoseInferencer initialized.")
#     return INFERENCER

_inferencer = None

def get_inferencer():
    global _inferencer
    if _inferencer is None:
        print("Initializing MMPoseInferencer with pose model: td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192")
        print("And detection model (using MMDetection alias/config)")
        try:
            _inferencer = MMPoseInferencer(
                pose2d='td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192', # 姿态模型
                # --- 修改这里 ---
                # 尝试使用 MMDetection 的标准模型名称或配置文件路径
                # 1. 尝试使用模型别名 (如果 MMPoseInferencer 支持)
                det_model='rtmdet-m', # 或者 'rtmdet_m_8xb32-100e_coco-obj365-person' (需要确认MMDet中准确的名称)
                # 2. 或者提供 MMDetection 配置文件的相对路径或绝对路径 (如果知道)
                # det_model='configs/rtmdet/rtmdet_m_8xb32-100e_coco-obj365-person.py', # 假设在MMDet的configs目录下
                # 权重会自动下载或从本地缓存加载 (如果模型名称已知)

                # --- 权重文件可以单独指定，或者让 MMDetection 根据模型名称自动处理 ---
                # det_weights='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',

                det_cat_ids=[0] # 目标检测类别 ID, 0 通常是 COCO 中的 'person'
            )
            print("MMPoseInferencer initialized successfully.")
        except Exception as e:
            print(f"!!! Error during MMPoseInferencer initialization: {e}") # 更明确的错误日志
            raise # 重新抛出异常，以便上层能捕获或 FastAPI 能记录
    return _inferencer

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

    # ---- DEBUGGING: PRINT THE STRUCTURE ----
    if results:
        print("Structure of the first frame's result (or first element of results):")
        try:
            import json
            # 尝试只打印 results[0] 的一部分，或者只打印键，避免直接序列化复杂对象
            if isinstance(results[0], dict):
                print(f"Keys in results[0]: {list(results[0].keys())}")
                if 'predictions' in results[0]:
                    print(f"Type of results[0]['predictions']: {type(results[0]['predictions'])}")
                    if results[0]['predictions'] and isinstance(results[0]['predictions'][0], list) and \
                            results[0]['predictions'][0]:
                        print(
                            f"Keys in first prediction instance: {list(results[0]['predictions'][0][0].keys()) if isinstance(results[0]['predictions'][0][0], dict) else 'Not a dict'}")
                # print(json.dumps(results[0], indent=2)) # 暂时注释掉这个，以防它出错
            else:
                print(f"results[0] is not a dict, it's a {type(results[0])}")

        except TypeError as e_json:
            print(f"ERROR: Failed to json.dumps results[0]. Error: {e_json}")
            print("Printing results[0] directly (might be large or complex):")
            print(results[0])  # 直接打印，看原始结构
        except Exception as e_print:
            print(f"ERROR during debug printing: {e_print}")
    # ---- END DEBUGGING ----

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