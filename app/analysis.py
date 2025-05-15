import numpy as np


def calculate_joint_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    return angle


def simple_compare_poses(pose_data_user, pose_data_target, frame_id=0):
    """
    A very basic comparison for a single frame.
    Compares specific joint angles.
    Assumes keypoints are [x, y, score]. We'll ignore score for angle calculation.
    COCO Keypoint indices (example, may vary slightly with model):
    0: nose, 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    if not pose_data_user or not pose_data_target:
        return {"error": "Missing pose data."}

    user_kps_frame = next((f for f in pose_data_user.get("pose_data", []) if f["frame_id"] == frame_id), None)
    target_kps_frame = next((f for f in pose_data_target.get("pose_data", []) if f["frame_id"] == frame_id), None)

    if not user_kps_frame or not target_kps_frame or not user_kps_frame["keypoints"] or not target_kps_frame[
        "keypoints"]:
        return {"error": f"Missing keypoints for frame {frame_id} in one or both poses."}

    user_kps = user_kps_frame["keypoints"]
    target_kps = target_kps_frame["keypoints"]

    # Ensure enough keypoints are detected (MMPose COCO usually has 17)
    if len(user_kps) < 17 or len(target_kps) < 17:
        return {"error": "Not enough keypoints detected for comparison (needs at least 17 for COCO)."}

    feedback = []

    # Example: Compare Left Elbow Angle (shoulder-elbow-wrist)
    # Keypoint indices: left_shoulder (5), left_elbow (7), left_wrist (9)
    try:
        user_le_angle = calculate_joint_angle(user_kps[5][:2], user_kps[7][:2], user_kps[9][:2])
        target_le_angle = calculate_joint_angle(target_kps[5][:2], target_kps[7][:2], target_kps[9][:2])
        diff_le_angle = user_le_angle - target_le_angle
        feedback.append(
            f"Left Elbow Angle: Your: {user_le_angle:.1f}°, Target: {target_le_angle:.1f}°, Diff: {diff_le_angle:.1f}°")
        if abs(diff_le_angle) > 15:  # Arbitrary threshold
            feedback.append(
                f"  Suggestion: Adjust your left elbow bend. {'Increase' if diff_le_angle < 0 else 'Decrease'} bending.")
    except Exception as e:
        feedback.append(f"Could not calculate Left Elbow Angle: {e}")

    # Example: Compare Left Knee Angle (hip-knee-ankle)
    # Keypoint indices: left_hip (11), left_knee (13), left_ankle (15)
    try:
        user_lk_angle = calculate_joint_angle(user_kps[11][:2], user_kps[13][:2], user_kps[15][:2])
        target_lk_angle = calculate_joint_angle(target_kps[11][:2], target_kps[13][:2], target_kps[15][:2])
        diff_lk_angle = user_lk_angle - target_lk_angle
        feedback.append(
            f"Left Knee Angle: Your: {user_lk_angle:.1f}°, Target: {target_lk_angle:.1f}°, Diff: {diff_lk_angle:.1f}°")
        if abs(diff_lk_angle) > 20:  # Arbitrary threshold
            feedback.append(
                f"  Suggestion: Adjust your left knee bend. {'Increase' if diff_lk_angle < 0 else 'Decrease'} bending.")
    except Exception as e:
        feedback.append(f"Could not calculate Left Knee Angle: {e}")

    if not feedback:
        feedback.append("No specific issues found with these basic checks, or comparison failed.")

    return {"feedback": feedback}