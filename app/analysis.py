import math
import numpy as np
from dtw import dtw # 你可能需要 pip install dtw-python


def get_angle(p1, p2, p3):
    """计算由三个点p1-p2-p3形成的夹角 (p2是顶点)"""
    # p1, p2, p3 都是 [x, y] 或 [x, y, score]
    v1 = np.array(p1[:2]) - np.array(p2[:2])
    v2 = np.array(p3[:2]) - np.array(p2[:2])
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Avoid division by zero
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to avoid domain errors
    return np.degrees(angle)


def get_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))


# Keypoint indices for COCO (assuming 17 keypoints, 0-indexed)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
KP_NAME_MAP = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


# def simple_compare_poses(user_pose_full, target_pose_full, frame_id=0):
#     feedback = []
#
#     # 从完整数据中提取指定帧的 keypoints
#     # user_pose_full 和 target_pose_full 的结构是:
#     # {"video_info": ..., "pose_data": [{"frame_id":0, "keypoints": [...]}, ...]}
#     # 或者 {"image_info": ..., "pose_data": [{"frame_id":0, "keypoints": [...]}]}
#
#     def get_kps_for_frame(pose_data_obj, f_id):
#         if not pose_data_obj or not pose_data_obj.get("pose_data"):
#             return None
#
#         # 如果是图像，pose_data 只有一个元素，frame_id 总是 0
#         is_image = "image_info" in pose_data_obj
#         actual_f_id = 0 if is_image else f_id
#
#         for frame in pose_data_obj["pose_data"]:
#             if frame["frame_id"] == actual_f_id:
#                 return frame.get("keypoints")
#         return None
#
#     user_kps = get_kps_for_frame(user_pose_full, frame_id)
#     target_kps = get_kps_for_frame(target_pose_full, frame_id)
#
#     if not user_kps or len(user_kps) == 0:
#         return {"error": f"User keypoints not found or empty for frame {frame_id}."}
#     if not target_kps or len(target_kps) == 0:
#         return {"error": f"Target keypoints not found or empty for frame {frame_id}."}
#
#     if len(user_kps) != 17 or len(target_kps) != 17:  # Assuming 17 COCO keypoints
#         return {"error": "Keypoint count mismatch (expected 17). Cannot compare reliably."}
#
#     # --- 1. 肢体角度比较 ---
#     angles_to_compare = {
#         "Left Elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
#         "Right Elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
#         "Left Knee": (L_HIP, L_KNEE, L_ANKLE),
#         "Right Knee": (R_HIP, R_KNEE, R_ANKLE),
#         "Left Shoulder (arm-torso)": (L_ELBOW, L_SHOULDER, L_HIP),  # 示例：手臂与躯干的角度
#         "Right Shoulder (arm-torso)": (R_ELBOW, R_SHOULDER, R_HIP),
#     }
#
#     angle_threshold = 20  # 角度差异阈值 (度)
#     feedback.append("--- Angle Comparison ---")
#     for name, (p1_idx, p2_idx, p3_idx) in angles_to_compare.items():
#         # 确保所有点都有足够高的置信度
#         if all(user_kps[i][2] > 0.3 for i in [p1_idx, p2_idx, p3_idx]) and \
#                 all(target_kps[i][2] > 0.3 for i in [p1_idx, p2_idx, p3_idx]):
#
#             user_angle = get_angle(user_kps[p1_idx], user_kps[p2_idx], user_kps[p3_idx])
#             target_angle = get_angle(target_kps[p1_idx], target_kps[p2_idx], target_kps[p3_idx])
#             diff = abs(user_angle - target_angle)
#
#             feedback.append(f"{name}: User={user_angle:.1f}°, Target={target_angle:.1f}°, Diff={diff:.1f}°")
#             if diff > angle_threshold:
#                 feedback.append(f"  - Consider adjusting your {name.lower()} angle.")
#         else:
#             feedback.append(f"{name}: Could not reliably calculate due to low keypoint confidence.")
#
#     # --- 2. 相对距离比较 (示例：双手间距) ---
#     # 需要归一化距离，例如用肩宽或身高作为参考，否则绝对像素距离意义不大
#     # 这里简单比较，不进行归一化，后续可以改进
#     feedback.append("\n--- Relative Distance Comparison (Example) ---")
#     if user_kps[L_WRIST][2] > 0.3 and user_kps[R_WRIST][2] > 0.3 and \
#             target_kps[L_WRIST][2] > 0.3 and target_kps[R_WRIST][2] > 0.3:
#         user_hand_dist = get_distance(user_kps[L_WRIST], user_kps[R_WRIST])
#         target_hand_dist = get_distance(target_kps[L_WRIST], target_kps[R_WRIST])
#
#         # 尝试用肩宽归一化
#         user_shoulder_width = get_distance(user_kps[L_SHOULDER], user_kps[R_SHOULDER])
#         target_shoulder_width = get_distance(target_kps[L_SHOULDER], target_kps[R_SHOULDER])
#
#         if user_shoulder_width > 1e-3 and target_shoulder_width > 1e-3:  # 避免除零
#             user_norm_hand_dist = user_hand_dist / user_shoulder_width
#             target_norm_hand_dist = target_hand_dist / target_shoulder_width
#             dist_diff_ratio = abs(user_norm_hand_dist - target_norm_hand_dist)
#
#             feedback.append(
#                 f"Normalized Hand Distance (rel. to shoulder width): User={user_norm_hand_dist:.2f}, Target={target_norm_hand_dist:.2f}")
#             if dist_diff_ratio > 0.3:  # 示例阈值
#                 feedback.append(f"  - The relative distance between your hands seems different from the target.")
#         else:
#             feedback.append("Hand Distance: Could not normalize due to unreliable shoulder keypoints.")
#     else:
#         feedback.append("Hand Distance: Could not reliably calculate due to low wrist keypoint confidence.")
#
#     # --- 3. 姿态对称性 (示例：双肩高度) ---
#     feedback.append("\n--- Symmetry Check (Example) ---")
#     if user_kps[L_SHOULDER][2] > 0.3 and user_kps[R_SHOULDER][2] > 0.3:
#         user_shoulder_y_diff = abs(user_kps[L_SHOULDER][1] - user_kps[R_SHOULDER][1])
#         # 同样，这个差异也需要归一化才有意义，例如相对于身高或躯干长度
#         # 这里简单给出绝对差异，提示用户注意
#         feedback.append(f"User Shoulder Y-Difference (pixels): {user_shoulder_y_diff:.1f}")
#         if user_shoulder_y_diff > 10:  # 像素差异阈值，非常粗略
#             feedback.append(f"  - Your shoulders might not be level. Compare with target's shoulder level.")
#
#     # --- 4. 更多维度可以逐步添加 ---
#     # 例如：躯干倾斜度
#     #       双脚间距
#     #       整体姿态的包围盒大小比较 (归一化后)
#
#     return {"feedback": feedback}


# def calculate_pose_vector_distance(pose_vec1, pose_vec2, distance_metric='euclidean'):
#     """计算两个单帧姿态向量之间的距离"""
#     if distance_metric == 'euclidean':
#         return np.linalg.norm(pose_vec1 - pose_vec2)
#     # 可以添加其他距离度量
#     return np.linalg.norm(pose_vec1 - pose_vec2) # 默认欧氏距离

def calculate_pose_vector_distance(pose_vec1, pose_vec2): # 单帧姿态向量距离
    return np.linalg.norm(np.array(pose_vec1) - np.array(pose_vec2))

def extract_flat_vectors_from_pose_sequence(pose_data_list):
    """从姿态序列中提取扁平化的(x,y)向量列表，用于DTW"""
    print(f"DEBUG: pose_data_list type: {type(pose_data_list)}")  # 看看它是不是列表
    if isinstance(pose_data_list, list) and pose_data_list:
        print(f"DEBUG: First element of pose_data_list type: {type(pose_data_list[0])}")
        print(f"DEBUG: First element content: {pose_data_list[0]}")
    # ... rest of the function

    vectors = []
    original_indices = [] # 记录原始帧的索引，方便回溯
    for idx, frame_data in enumerate(pose_data_list):
        if frame_data and frame_data.get("keypoints") and len(frame_data["keypoints"]) == 17:
            vec = []
            valid_frame = True
            # 确保所有关键点置信度不太低，或者设定一个阈值来决定是否使用该帧
            # for kp in frame_data["keypoints"]:
            #     if kp[2] < 0.1: # 示例：极低置信度的帧可能不参与DTW
            #         valid_frame = False
            #         break
            # if not valid_frame:
            #     continue

            for kp in frame_data["keypoints"]: # kp is [x, y, score]
                vec.extend(kp[:2]) # 只取 x, y
            if vec:
                vectors.append(np.array(vec))
                original_indices.append(frame_data.get("frame_id", idx)) # 优先用frame_id
        # else:
            # print(f"Skipping frame {idx} due to missing/incomplete keypoints for DTW.")
    return vectors, original_indices


def compare_pose_sequences_dtw_detailed(target_pose_sequence_full, user_pose_sequence_full):
    """
    使用DTW比较两个姿态序列，并返回详细的对齐和差异信息。
    *_pose_sequence_full: 完整的pose数据对象，包含 "pose_data" 列表
    """

    # 从 target_pose_sequence_full (即 TARGET_VIDEO_DATA) 中提取真正的帧列表
    target_pose_results_dict = target_pose_sequence_full.get("pose_data")
    if target_pose_results_dict and isinstance(target_pose_results_dict, dict):
        target_poses_list = target_pose_results_dict.get("pose_data", [])
    else:
        target_poses_list = []
        # print("Warning: 'pose_data' key in target_pose_sequence_full did not yield the expected dictionary.")

    # 从 user_pose_sequence_full (即 USER_VIDEO_DATA) 中提取真正的帧列表
    user_pose_results_dict = user_pose_sequence_full.get("pose_data")
    if user_pose_results_dict and isinstance(user_pose_results_dict, dict):
        user_poses_list = user_pose_results_dict.get("pose_data", [])
    else:
        user_poses_list = []
        # print("Warning: 'pose_data' key in user_pose_sequence_full did not yield the expected dictionary.")

    # 调试打印
    # print(f"DEBUG: target_poses_list is now: {type(target_poses_list)}")
    # if target_poses_list: print(f"DEBUG: first element of target_poses_list is: {type(target_poses_list[0])}")

    if not target_poses_list or not user_poses_list:
        return {"error": "One or both pose sequences for DTW are empty after extraction."}

    target_vectors, target_original_indices = extract_flat_vectors_from_pose_sequence(target_poses_list)
    user_vectors, user_original_indices = extract_flat_vectors_from_pose_sequence(user_poses_list)

    if not target_vectors or not user_vectors:
        return {"error": "Failed to extract feature vectors from one or both sequences for DTW."}

    if target_vectors[0].shape != user_vectors[0].shape:
        return {
            "error": f"Pose vector dimensions mismatch for DTW: {target_vectors[0].shape} vs {user_vectors[0].shape}"}

    try:
        alignment = dtw(target_vectors, user_vectors,
                        keep_internals=True,
                        dist_method=calculate_pose_vector_distance)  # 使用欧氏距离

        dtw_distance = alignment.distance
        normalized_dtw_distance = alignment.normalizedDistance

        # alignment.index1 对应 target_vectors 的索引
        # alignment.index2 对应 user_vectors 的索引
        aligned_frame_pairs_details = []
        # 用于存储每对对齐帧的差异得分，以便找出差异最大的帧
        frame_pair_diff_scores = []

        for i in range(len(alignment.index1)):
            target_vec_idx = alignment.index1[i]
            user_vec_idx = alignment.index2[i]

            # 通过 vector 索引找到原始的 frame_id
            original_target_frame_id = target_original_indices[target_vec_idx]
            original_user_frame_id = user_original_indices[user_vec_idx]

            # 获取这两帧的详细姿态数据以进行比较
            # 注意: simple_compare_poses 需要的是完整的 pose_data 对象和 frame_id
            # 我们需要传递原始的 *_pose_sequence_full 和原始 frame_id
            # 或者，可以直接传递这两帧的keypoints给一个修改版的simple_compare_poses

            # 传递原始的完整数据对象和 frame_id
            # simple_compare_poses 内部会根据 frame_id 查找 keypoints
            detailed_comparison = simple_compare_poses(user_pose_sequence_full,
                                                       target_pose_sequence_full,
                                                       frame_id_user=original_user_frame_id,
                                                       # 需要修改 simple_compare_poses 接受两个frame_id
                                                       frame_id_target=original_target_frame_id)

            # 为了简化，我们先计算一个简单的差异得分，例如基于对齐向量的距离
            current_pair_diff = calculate_pose_vector_distance(target_vectors[target_vec_idx],
                                                               user_vectors[user_vec_idx])
            frame_pair_diff_scores.append({
                "target_frame_id": original_target_frame_id,
                "user_frame_id": original_user_frame_id,
                "difference_score": current_pair_diff,  # 数值差异
                "detailed_feedback": detailed_comparison.get("feedback",
                                                             ["No detailed feedback."]) if "error" not in detailed_comparison else [
                    detailed_comparison["error"]]
            })

        # 根据 difference_score 排序，找出差异最大的N帧
        # (如果序列很长，只返回差异最大的几帧的详细feedback)
        frame_pair_diff_scores.sort(key=lambda x: x["difference_score"], reverse=True)

        top_n_diff_frames = 5  # 返回差异最大的5帧
        most_different_frames_feedback = frame_pair_diff_scores[:top_n_diff_frames]

        return {
            "dtw_distance": dtw_distance,
            "normalized_dtw_distance": normalized_dtw_distance,
            "alignment_path_target_indices": alignment.index1.tolist(),  # target_vectors中的索引
            "alignment_path_user_indices": alignment.index2.tolist(),  # user_vectors中的索引
            "original_target_frame_ids_in_path": [target_original_indices[i] for i in alignment.index1],
            "original_user_frame_ids_in_path": [user_original_indices[i] for i in alignment.index2],
            "most_different_frames_feedback": most_different_frames_feedback,
            "message": "DTW detailed comparison successful."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"DTW detailed calculation failed: {str(e)}"}


def simple_compare_poses(user_pose_full, target_pose_full, frame_id_user=0, frame_id_target=0):
    feedback = []

    def get_kps_for_frame(full_pose_data_object, f_id):
        # full_pose_data_object 是 TARGET_VIDEO_DATA 或 USER_VIDEO_DATA
        # 它包含一个 "pose_data" 键，其值是 { "video_info": ..., "pose_data": [frames_list] }

        # 1. 获取内层的 pose_results 字典
        pose_results_dict = full_pose_data_object.get("pose_data")
        if not pose_results_dict or not isinstance(pose_results_dict, dict):
            print(
                f"Warning in get_kps_for_frame: pose_results_dict not found or not a dict. Object was: {full_pose_data_object.get('type')}")
            return None

        # 2. 从 pose_results_dict 中获取真正的帧列表
        frames_list = pose_results_dict.get("pose_data")
        if not frames_list or not isinstance(frames_list, list):
            print(f"Warning in get_kps_for_frame: frames_list not found or not a list inside pose_results_dict.")
            return None

        # 3. 判断是图像还是视频，以确定实际使用的 frame_id
        #    这个信息可以从 full_pose_data_object 的 "type" 键获取，或者检查 pose_results_dict 里是否有 "image_info"
        media_type = full_pose_data_object.get("type")  # 'video' or 'image'
        actual_f_id = 0 if media_type == "image" else f_id

        # 4. 在帧列表中查找
        for frame in frames_list:
            if isinstance(frame, dict) and frame.get("frame_id") == actual_f_id:
                return frame.get("keypoints")

        print(f"Warning in get_kps_for_frame: Frame {actual_f_id} not found in frames_list.")
        return None

    user_kps = get_kps_for_frame(user_pose_full, frame_id_user)
    target_kps = get_kps_for_frame(target_pose_full, frame_id_target)

    # ... (其余的 simple_compare_poses 逻辑) ...
    # 例如:
    if not user_kps or len(user_kps) == 0:
        return {"error": f"User keypoints not found or empty for frame {frame_id_user}."}
    if not target_kps or len(target_kps) == 0:
        return {"error": f"Target keypoints not found or empty for frame {frame_id_target}."}

    if len(user_kps) != 17 or len(target_kps) != 17:
        return {"error": "Keypoint count mismatch (expected 17)."}

    feedback.append(f"--- Comparing Target Frame {frame_id_target} with User Frame {frame_id_user} ---")
    angles_to_compare = {
        "Left Elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
        "Right Elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
    }
    angle_threshold = 20
    for name, (p1_idx, p2_idx, p3_idx) in angles_to_compare.items():
        # 确保所有点都有足够高的置信度
        # 并且 user_kps 和 target_kps 都是有效的列表
        if isinstance(user_kps, list) and isinstance(target_kps, list) and \
                all(isinstance(user_kps[i], list) and len(user_kps[i]) > 2 and user_kps[i][2] > 0.3 for i in
                    [p1_idx, p2_idx, p3_idx]) and \
                all(isinstance(target_kps[i], list) and len(target_kps[i]) > 2 and target_kps[i][2] > 0.3 for i in
                    [p1_idx, p2_idx, p3_idx]):

            user_angle = get_angle(user_kps[p1_idx], user_kps[p2_idx], user_kps[p3_idx])
            target_angle = get_angle(target_kps[p1_idx], target_kps[p2_idx], target_kps[p3_idx])
            diff = abs(user_angle - target_angle)

            feedback.append(f"  {name}: User={user_angle:.1f}°, Target={target_angle:.1f}°, Diff={diff:.1f}°")
            if diff > angle_threshold:
                feedback.append(f"    - Consider adjusting your {name.lower()} angle.")
        else:
            feedback.append(f"  {name}: Low keypoint confidence or invalid keypoint data.")

    return {"feedback": feedback}


# # 需要修改 simple_compare_poses 来接受两个不同的 frame_id
# def simple_compare_poses(user_pose_full, target_pose_full, frame_id_user=0, frame_id_target=0):
#     feedback = []
#
#     def get_kps_for_frame(pose_data_obj, f_id):
#         # ... (这个函数保持不变)
#         if not pose_data_obj or not pose_data_obj.get("pose_data"): return None
#         is_image = "image_info" in pose_data_obj
#         actual_f_id = 0 if is_image else f_id
#         for frame in pose_data_obj["pose_data"]:
#             if frame["frame_id"] == actual_f_id: return frame.get("keypoints")
#         return None
#
#     user_kps = get_kps_for_frame(user_pose_full, frame_id_user)
#     target_kps = get_kps_for_frame(target_pose_full, frame_id_target)
#
#     # ... (其余的 simple_compare_poses 逻辑保持不变, 使用 user_kps 和 target_kps)
#     if not user_kps or len(user_kps) == 0:
#         return {"error": f"User keypoints not found or empty for frame {frame_id_user}."}
#     if not target_kps or len(target_kps) == 0:
#         return {"error": f"Target keypoints not found or empty for frame {frame_id_target}."}
#
#     if len(user_kps) != 17 or len(target_kps) != 17:
#         return {"error": "Keypoint count mismatch (expected 17)."}
#
#     # --- Angle Comparison --- (示例)
#     feedback.append(f"--- Comparing Target Frame {frame_id_target} with User Frame {frame_id_user} ---")
#     angles_to_compare = {
#         "Left Elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
#         "Right Elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
#     }
#     angle_threshold = 20
#     for name, (p1_idx, p2_idx, p3_idx) in angles_to_compare.items():
#         if all(user_kps[i][2] > 0.3 for i in [p1_idx, p2_idx, p3_idx]) and \
#                 all(target_kps[i][2] > 0.3 for i in [p1_idx, p2_idx, p3_idx]):
#             user_angle = get_angle(user_kps[p1_idx], user_kps[p2_idx], user_kps[p3_idx])
#             target_angle = get_angle(target_kps[p1_idx], target_kps[p2_idx], target_kps[p3_idx])
#             diff = abs(user_angle - target_angle)
#             feedback.append(f"  {name}: User={user_angle:.1f}°, Target={target_angle:.1f}°, Diff={diff:.1f}°")
#             if diff > angle_threshold:
#                 feedback.append(f"    - Consider adjusting your {name.lower()} angle.")
#         else:
#             feedback.append(f"  {name}: Low keypoint confidence.")
#
#     # ... (可以添加更多 simple_compare_poses 的比较维度) ...
#
#     return {"feedback": feedback}

def compare_pose_sequences_dtw(sequence1_poses_data, sequence2_poses_data):
    """
    使用DTW比较两个姿态序列。
    sequence_poses_data: 列表，每个元素是一帧的姿态数据，例如
                         [{"frame_id": 0, "keypoints": [[x,y,score],...]}, ...]
    """
    if not sequence1_poses_data or not sequence2_poses_data:
        return {"error": "One or both pose sequences are empty."}

    # 1. 将关键点转换为向量序列
    def extract_vectors(pose_data_list):
        vectors = []
        for frame_data in pose_data_list:
            if frame_data and frame_data.get("keypoints"):
                # 扁平化 keypoints (只取 x, y)，忽略 score 或使用 score 加权
                # 确保所有帧的关键点数量一致，或进行填充/截断处理
                # 这里简单地将所有 x, y 连起来
                vec = []
                for kp in frame_data["keypoints"]:  # kp is [x, y, score]
                    vec.extend(kp[:2])  # 只取 x, y
                if vec:  # 确保不是空列表
                    vectors.append(np.array(vec))
        return vectors

    seq1_vectors = extract_vectors(sequence1_poses_data)
    seq2_vectors = extract_vectors(sequence2_poses_data)

    if not seq1_vectors or not seq2_vectors:
        return {"error": "Failed to extract feature vectors from one or both sequences."}

    # 确保向量维度一致，这里假设它们应该是一致的
    # 如果第一帧的维度不一致，可能后续都会有问题
    if seq1_vectors[0].shape != seq2_vectors[0].shape:
        return {"error": f"Pose vector dimensions mismatch: {seq1_vectors[0].shape} vs {seq2_vectors[0].shape}"}

    # 2. 计算DTW
    # dtw-python 的 dtw 函数需要一个自定义的距离函数 (lambda)
    # 或者你可以预先计算所有帧对之间的距离矩阵
    try:
        # 使用 lambda 定义距离函数，这里的 x, y 是序列中的两个姿态向量
        alignment = dtw(seq1_vectors, seq2_vectors,
                        keep_internals=True,
                        dist_method=calculate_pose_vector_distance)  # 或者直接用 'euclidean' 如果库支持

        dtw_distance = alignment.distance  # 累积距离
        normalized_dtw_distance = alignment.normalizedDistance  # 归一化距离（更常用）

        # path 返回的是对齐的索引对 (index_seq1, index_seq2)
        # path_indices = alignment.index1, alignment.index2

        return {
            "dtw_distance": dtw_distance,
            "normalized_dtw_distance": normalized_dtw_distance,
            "message": "DTW comparison successful."
            # "alignment_path_length": len(path_indices[0]) # 对齐路径的长度
        }
    except Exception as e:
        return {"error": f"DTW calculation failed: {str(e)}"}


def calculate_joint_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    return angle


# def simple_compare_poses(pose_data_user, pose_data_target, frame_id=0):
#     """
#     A very basic comparison for a single frame.
#     Compares specific joint angles.
#     Assumes keypoints are [x, y, score]. We'll ignore score for angle calculation.
#     COCO Keypoint indices (example, may vary slightly with model):
#     0: nose, 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
#     9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
#     13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
#     """
#     if not pose_data_user or not pose_data_target:
#         return {"error": "Missing pose data."}
#
#     user_kps_frame = next((f for f in pose_data_user.get("pose_data", []) if f["frame_id"] == frame_id), None)
#     target_kps_frame = next((f for f in pose_data_target.get("pose_data", []) if f["frame_id"] == frame_id), None)
#
#     if not user_kps_frame or not target_kps_frame or not user_kps_frame["keypoints"] or not target_kps_frame[
#         "keypoints"]:
#         return {"error": f"Missing keypoints for frame {frame_id} in one or both poses."}
#
#     user_kps = user_kps_frame["keypoints"]
#     target_kps = target_kps_frame["keypoints"]
#
#     # Ensure enough keypoints are detected (MMPose COCO usually has 17)
#     if len(user_kps) < 17 or len(target_kps) < 17:
#         return {"error": "Not enough keypoints detected for comparison (needs at least 17 for COCO)."}
#
#     feedback = []
#
#     # Example: Compare Left Elbow Angle (shoulder-elbow-wrist)
#     # Keypoint indices: left_shoulder (5), left_elbow (7), left_wrist (9)
#     try:
#         user_le_angle = calculate_joint_angle(user_kps[5][:2], user_kps[7][:2], user_kps[9][:2])
#         target_le_angle = calculate_joint_angle(target_kps[5][:2], target_kps[7][:2], target_kps[9][:2])
#         diff_le_angle = user_le_angle - target_le_angle
#         feedback.append(
#             f"Left Elbow Angle: Your: {user_le_angle:.1f}°, Target: {target_le_angle:.1f}°, Diff: {diff_le_angle:.1f}°")
#         if abs(diff_le_angle) > 15:  # Arbitrary threshold
#             feedback.append(
#                 f"  Suggestion: Adjust your left elbow bend. {'Increase' if diff_le_angle < 0 else 'Decrease'} bending.")
#     except Exception as e:
#         feedback.append(f"Could not calculate Left Elbow Angle: {e}")
#
#     # Example: Compare Left Knee Angle (hip-knee-ankle)
#     # Keypoint indices: left_hip (11), left_knee (13), left_ankle (15)
#     try:
#         user_lk_angle = calculate_joint_angle(user_kps[11][:2], user_kps[13][:2], user_kps[15][:2])
#         target_lk_angle = calculate_joint_angle(target_kps[11][:2], target_kps[13][:2], target_kps[15][:2])
#         diff_lk_angle = user_lk_angle - target_lk_angle
#         feedback.append(
#             f"Left Knee Angle: Your: {user_lk_angle:.1f}°, Target: {target_lk_angle:.1f}°, Diff: {diff_lk_angle:.1f}°")
#         if abs(diff_lk_angle) > 20:  # Arbitrary threshold
#             feedback.append(
#                 f"  Suggestion: Adjust your left knee bend. {'Increase' if diff_lk_angle < 0 else 'Decrease'} bending.")
#     except Exception as e:
#         feedback.append(f"Could not calculate Left Knee Angle: {e}")
#
#     if not feedback:
#         feedback.append("No specific issues found with these basic checks, or comparison failed.")
#
#     return {"feedback": feedback}