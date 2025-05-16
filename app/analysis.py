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


def simple_compare_poses(user_pose_full, target_pose_full, frame_id=0):
    feedback = []

    # 从完整数据中提取指定帧的 keypoints
    # user_pose_full 和 target_pose_full 的结构是:
    # {"video_info": ..., "pose_data": [{"frame_id":0, "keypoints": [...]}, ...]}
    # 或者 {"image_info": ..., "pose_data": [{"frame_id":0, "keypoints": [...]}]}

    def get_kps_for_frame(pose_data_obj, f_id):
        if not pose_data_obj or not pose_data_obj.get("pose_data"):
            return None

        # 如果是图像，pose_data 只有一个元素，frame_id 总是 0
        is_image = "image_info" in pose_data_obj
        actual_f_id = 0 if is_image else f_id

        for frame in pose_data_obj["pose_data"]:
            if frame["frame_id"] == actual_f_id:
                return frame.get("keypoints")
        return None

    user_kps = get_kps_for_frame(user_pose_full, frame_id)
    target_kps = get_kps_for_frame(target_pose_full, frame_id)

    if not user_kps or len(user_kps) == 0:
        return {"error": f"User keypoints not found or empty for frame {frame_id}."}
    if not target_kps or len(target_kps) == 0:
        return {"error": f"Target keypoints not found or empty for frame {frame_id}."}

    if len(user_kps) != 17 or len(target_kps) != 17:  # Assuming 17 COCO keypoints
        return {"error": "Keypoint count mismatch (expected 17). Cannot compare reliably."}

    # --- 1. 肢体角度比较 ---
    angles_to_compare = {
        "Left Elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
        "Right Elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
        "Left Knee": (L_HIP, L_KNEE, L_ANKLE),
        "Right Knee": (R_HIP, R_KNEE, R_ANKLE),
        "Left Shoulder (arm-torso)": (L_ELBOW, L_SHOULDER, L_HIP),  # 示例：手臂与躯干的角度
        "Right Shoulder (arm-torso)": (R_ELBOW, R_SHOULDER, R_HIP),
    }

    angle_threshold = 20  # 角度差异阈值 (度)
    feedback.append("--- Angle Comparison ---")
    for name, (p1_idx, p2_idx, p3_idx) in angles_to_compare.items():
        # 确保所有点都有足够高的置信度
        if all(user_kps[i][2] > 0.3 for i in [p1_idx, p2_idx, p3_idx]) and \
                all(target_kps[i][2] > 0.3 for i in [p1_idx, p2_idx, p3_idx]):

            user_angle = get_angle(user_kps[p1_idx], user_kps[p2_idx], user_kps[p3_idx])
            target_angle = get_angle(target_kps[p1_idx], target_kps[p2_idx], target_kps[p3_idx])
            diff = abs(user_angle - target_angle)

            feedback.append(f"{name}: User={user_angle:.1f}°, Target={target_angle:.1f}°, Diff={diff:.1f}°")
            if diff > angle_threshold:
                feedback.append(f"  - Consider adjusting your {name.lower()} angle.")
        else:
            feedback.append(f"{name}: Could not reliably calculate due to low keypoint confidence.")

    # --- 2. 相对距离比较 (示例：双手间距) ---
    # 需要归一化距离，例如用肩宽或身高作为参考，否则绝对像素距离意义不大
    # 这里简单比较，不进行归一化，后续可以改进
    feedback.append("\n--- Relative Distance Comparison (Example) ---")
    if user_kps[L_WRIST][2] > 0.3 and user_kps[R_WRIST][2] > 0.3 and \
            target_kps[L_WRIST][2] > 0.3 and target_kps[R_WRIST][2] > 0.3:
        user_hand_dist = get_distance(user_kps[L_WRIST], user_kps[R_WRIST])
        target_hand_dist = get_distance(target_kps[L_WRIST], target_kps[R_WRIST])

        # 尝试用肩宽归一化
        user_shoulder_width = get_distance(user_kps[L_SHOULDER], user_kps[R_SHOULDER])
        target_shoulder_width = get_distance(target_kps[L_SHOULDER], target_kps[R_SHOULDER])

        if user_shoulder_width > 1e-3 and target_shoulder_width > 1e-3:  # 避免除零
            user_norm_hand_dist = user_hand_dist / user_shoulder_width
            target_norm_hand_dist = target_hand_dist / target_shoulder_width
            dist_diff_ratio = abs(user_norm_hand_dist - target_norm_hand_dist)

            feedback.append(
                f"Normalized Hand Distance (rel. to shoulder width): User={user_norm_hand_dist:.2f}, Target={target_norm_hand_dist:.2f}")
            if dist_diff_ratio > 0.3:  # 示例阈值
                feedback.append(f"  - The relative distance between your hands seems different from the target.")
        else:
            feedback.append("Hand Distance: Could not normalize due to unreliable shoulder keypoints.")
    else:
        feedback.append("Hand Distance: Could not reliably calculate due to low wrist keypoint confidence.")

    # --- 3. 姿态对称性 (示例：双肩高度) ---
    feedback.append("\n--- Symmetry Check (Example) ---")
    if user_kps[L_SHOULDER][2] > 0.3 and user_kps[R_SHOULDER][2] > 0.3:
        user_shoulder_y_diff = abs(user_kps[L_SHOULDER][1] - user_kps[R_SHOULDER][1])
        # 同样，这个差异也需要归一化才有意义，例如相对于身高或躯干长度
        # 这里简单给出绝对差异，提示用户注意
        feedback.append(f"User Shoulder Y-Difference (pixels): {user_shoulder_y_diff:.1f}")
        if user_shoulder_y_diff > 10:  # 像素差异阈值，非常粗略
            feedback.append(f"  - Your shoulders might not be level. Compare with target's shoulder level.")

    # --- 4. 更多维度可以逐步添加 ---
    # 例如：躯干倾斜度
    #       双脚间距
    #       整体姿态的包围盒大小比较 (归一化后)

    return {"feedback": feedback}


def calculate_pose_vector_distance(pose_vec1, pose_vec2, distance_metric='euclidean'):
    """计算两个单帧姿态向量之间的距离"""
    if distance_metric == 'euclidean':
        return np.linalg.norm(pose_vec1 - pose_vec2)
    # 可以添加其他距离度量
    return np.linalg.norm(pose_vec1 - pose_vec2) # 默认欧氏距离


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