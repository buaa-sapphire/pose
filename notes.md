好的，这是一个非常有趣且具有挑战性的项目！构建这样一个运动视频分析和指导系统，涉及到计算机视觉、深度学习、前后端开发等多个领域。下面我将详细分析所需技术、模型，并给出一个基础的实现思路和代码框架。

**项目目标：**

1.  用户上传目标运动员视频（作为参考标准）。
2.  系统对目标视频进行分析，提取关键运动姿态/序列。
3.  用户上传自己的运动视频或图片。
4.  系统分析用户视频/图片，与目标视频的姿态进行对比。
5.  系统指出用户动作的潜在问题，并给出纠正建议。

**核心技术分析**

1.  **人体姿态估计 (Human Pose Estimation)**
    *   **作用**：这是项目的核心。从视频/图片中检测人体的关键点（如头、肩、肘、腕、髋、膝、踝等）。
    *   **开源框架/模型**：
        *   **MMPose (OpenMMLab)**: 强烈推荐。它是一个基于 PyTorch 的姿态估计算法库，支持众多主流的 2D/3D 姿态估计模型（如 HRNet, SimpleBaseline, ViTPose 等）。模型丰富，预训练权重多，社区活跃。
        *   **MediaPipe Pose (Google)**: 轻量级，跨平台，实时性好。对于快速原型验证和一些不需要极致精度的场景很合适。
        *   **AlphaPose**: 另一个优秀的姿态估计框架。
    *   **选择理由**：MMPose 提供了强大的模型和灵活性，适合需要较高精度的分析。4090 显卡可以很好地支持其复杂模型进行推理，甚至进行微调（如果需要）。

2.  **动作识别/分类 (Action Recognition/Classification) - 可选，但推荐**
    *   **作用**：如果运动类型多样（骑行、篮球、滑雪），先识别出具体的动作（如投篮、运球、蹬踏、滑降转弯）可以帮助系统调用更针对性的分析模块和参考标准。
    *   **开源框架/模型**：
        *   **MMAction2 (OpenMMLab)**: 与 MMPose 同属 OpenMMLab 生态，基于 PyTorch，支持多种视频理解模型（如 TSN, SlowFast, Timesformer）。
        *   **VideoMAE, ViViT**: 基于 Transformer 的先进视频分类模型。
    *   **选择理由**：MMAction2 与 MMPose 结合紧密，可以共享一些底层库。

3.  **姿态/动作对比与评估**
    *   **作用**：这是实现“纠错”和“指导”的关键。
    *   **技术方案**：
        *   **关键点序列对齐**：
            *   **动态时间规整 (Dynamic Time Warping, DTW)**: 用于比较两个时间长度可能不同的姿态序列的相似度，并找到它们之间的最佳对齐方式。非常适合比较两个动作的节奏和形态差异。
            *   **Procrustes Analysis**: 用于对齐两组形状（关键点集合），消除平移、旋转和缩放的影响，从而比较纯粹的形状差异。
        *   **角度分析**：计算特定关节的角度（如膝关节弯曲度、肘关节角度），并与标准动作的角度范围进行比较。
        *   **轨迹分析**：分析特定关键点（如手、脚）的运动轨迹，与标准轨迹对比。
        *   **基于规则的评估**：根据运动学原理和专家知识，设定一系列评估规则。例如：“投篮时，肘部应高于肩部”，“深蹲时，膝盖不应超过脚尖太多”。
        *   **机器学习评估模型 (进阶)**：收集大量好/坏动作的样本，训练一个分类器或回归模型来直接评估动作质量或打分。这需要大量的标注数据。

4.  **视频/图像处理**
    *   **作用**：视频帧提取、图像缩放、裁剪、绘制骨骼等。
    *   **开源库**：
        *   **OpenCV (cv2)**: 强大的计算机视觉库，用于视频读写、图像处理、绘图等。
        *   **Pillow (PIL)**: 图像处理库。

5.  **数据存储**
    *   **作用**：存储上传的视频、提取的姿态数据、分析结果等。
    *   **方案**：
        *   **文件系统**: 直接存储视频文件。
        *   **JSON/CSV/Numpy 文件**: 存储姿态关键点序列、分析结果。
        *   **数据库 (SQLite, PostgreSQL)**: 如果需要更结构化的管理和查询用户数据、模型参考数据等。对于原型，文件系统 + JSON 可能足够。

6.  **后端框架**
    *   **作用**：接收前端请求，调用分析模型，返回结果。
    *   **开源框架**：
        *   **FastAPI**: 高性能，易于学习，基于 Python 类型提示，自动生成 API 文档。非常适合构建 ML 服务的 API。
        *   **Flask**: 轻量级，灵活，老牌框架。
    *   **选择理由**: FastAPI 因其现代特性和性能优势更受推荐。

7.  **前端显示**
    *   **作用**：用户上传界面，视频/图片展示，分析结果可视化（如骨骼叠加、问题高亮、文字指导）。
    *   **技术栈**：
        *   **HTML, CSS, JavaScript (Vanilla JS)**: 基础。
        *   **前端框架 (可选，但推荐用于复杂交互)**: Vue.js, React, Svelte。
        *   **图表/可视化库 (可选)**: Chart.js, D3.js (用于展示数据趋势), Konva.js (HTML5 Canvas 库，方便在视频上绘制)。
    *   **选择理由**: 对于一个包含前后端的完整代码示例，我会尽量使用 Vanilla JS 以减少依赖，但会指出使用框架的好处。

**模型选择和训练**

*   **姿态估计**: 使用 MMPose 中的 **HRNet** (如 `hrnet_w48_coco_256x192`) 或 **ViTPose** (如 `vitpose-base-coco-256x192`) 的预训练模型。这些模型在 COCO 等大规模数据集上预训练，泛化能力较好。你的 4090 显卡足以流畅运行这些模型的推理。
    *   **微调 (Fine-tuning)**: 如果特定运动（如滑雪的特殊姿态）在预训练数据集中覆盖不足，导致精度不够，可以考虑收集特定运动的标注数据进行微调。但这需要额外的数据收集和标注工作。
*   **动作识别 (如果使用)**: 使用 MMAction2 中的 **SlowFast** 或 **Timesformer** 的预训练模型（如在 Kinetics-400/600 数据集上预训练的）。
*   **训练**: 除非你要进行微调或训练自定义的评估模型，否则主要工作是使用预训练模型进行推理。你的 4090 显卡将主要用于加速推理过程。

**技术栈总结**

*   **语言**: Python 3.10.16
*   **环境管理**: Conda
*   **深度学习**: PyTorch (通过 MMPose, MMAction2)
*   **姿态估计**: MMPose (HRNet/ViTPose)
*   **动作识别**: MMAction2 (SlowFast/Timesformer) - 可选
*   **视频/图像处理**: OpenCV, Pillow
*   **数值计算**: NumPy
*   **后端**: FastAPI
*   **前端**: HTML, CSS, JavaScript (可考虑 p5.js 或 Konva.js 进行Canvas绘图)
*   **GPU 加速**: CUDA (通过 PyTorch)

**详细步骤和工作流程**

1.  **环境搭建**
    *   安装 Conda。
    *   创建 Conda 环境并安装 Python 3.10.16。
    *   安装 PyTorch (GPU 版本，与你的 CUDA 版本匹配)。
    *   安装 MMCV, MMEngine, MMPose, MMAction2 (如果使用)。
    *   安装 FastAPI, Uvicorn, OpenCV, NumPy, Pillow 等。

2.  **后端开发 (FastAPI)**
    *   **API 端点设计**:
        *   `/upload_target_video`: 上传目标运动员视频。
        *   `/upload_user_video`: 上传用户视频/图片。
        *   `/analyze_video`: (内部调用或由上传触发) 对视频进行姿态估计和可选的动作识别。
        *   `/get_comparison_result`: 获取对比分析结果和指导。
    *   **核心逻辑**:
        *   **视频处理模块**:
            *   读取视频，逐帧提取。
            *   调用 MMPose 进行姿态估计，获取每帧的关键点坐标。
            *   (可选) 调用 MMAction2 进行动作片段识别。
        *   **数据存储模块**:
            *   保存上传的视频。
            *   将提取的关键点序列保存为 JSON 或 NumPy 文件，与视频关联。
        *   **对比分析模块**:
            *   加载目标视频的关键点序列和用户视频的关键点序列。
            *   **对齐**: 使用 DTW 对齐两个姿态序列。
            *   **计算差异**:
                *   比较对应帧（对齐后）的关键点位置差异 (欧氏距离)。
                *   计算关键关节角度（如肘、膝、髋）的差异。
                *   (进阶) 比较关键点轨迹。
            *   **生成指导**:
                *   基于预设规则库：例如，如果用户膝关节弯曲度远小于目标，则提示“请增加膝关节弯曲度”。
                *   高亮显示差异较大的身体部位。
    *   **数据结构**:
        *   姿态数据: `[{frame_id: int, keypoints: [[x, y, conf], ...]}, ...]`

3.  **前端开发 (HTML, JS, CSS)**
    *   **界面元素**:
        *   视频上传控件 (目标视频、用户视频)。
        *   视频播放器 (用于展示原始视频和带骨骼叠加的视频)。
        *   结果展示区域 (文字指导、问题列表、对比图示)。
    *   **交互逻辑**:
        *   用户选择视频文件，通过 JavaScript `FormData` 异步上传到后端 API。
        *   上传成功后，轮询或通过 WebSocket 等待后端分析完成。
        *   获取分析结果 (JSON 格式，包含关键点数据、对比差异、指导文本)。
        *   在视频上绘制骨骼：使用 HTML5 Canvas，根据关键点坐标绘制线条和圆点。可以同时绘制用户和目标（半透明）的骨骼进行对比。
        *   将文字指导显示在页面上。

4.  **整合与测试**
    *   确保前后端能够正确通信。
    *   用不同类型的运动视频测试系统的鲁棒性和准确性。
    *   根据测试结果调整对比算法的参数和指导规则。

**代码实现框架 (简化版)**

由于提供一个“完整”且生产级的代码非常庞大，这里给出一个核心思路和简化的代码结构。
我们将主要关注：
1.  使用 MMPose 进行姿态估计。
2.  一个非常基础的后端 API (FastAPI)。
3.  一个极简的前端页面用于上传和显示结果（骨骼）。
4.  对比和指导部分会做简化，主要展示如何获取和使用姿态数据。

**1. Conda 环境准备**

```bash
# 创建 conda 环境
conda create -n sport_analysis python=3.10 -y
conda activate sport_analysis

# 安装 PyTorch (根据你的 CUDA 版本从官网选择合适的命令)
# 例如 CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装 OpenMMLab 相关库 (MIM 是 OpenMMLab 的包管理器)
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmpose

# 安装其他依赖
pip install fastapi uvicorn opencv-python numpy Pillow
pip install python-multipart # for FastAPI file uploads
```

**2. 项目结构 (示例)**

```
sport_analyzer/
├── app/
│   ├── main.py             # FastAPI 应用
│   ├── pose_estimator.py   # 姿态估计模块
│   ├── analysis.py         # 对比分析模块 (简化)
│   └── utils.py            # 工具函数
├── static/                 # 前端静态文件
│   ├── index.html
│   └── script.js
│   └── style.css
├── uploads/                # 存放上传的视频
├── models/                 # (如果需要下载模型文件到本地)
└── results/                # 存放分析结果 (如关键点json)
```

**3. 后端代码 (`app/`)**

**`app/pose_estimator.py`**

```python
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

MODEL_ALIAS = 'td-hm_ViTPose-base_8xb64-210e_coco-256x192' # 或者 'hrnet_w48_coco_256x192'
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
    result_generator = inferencer(video_path, return_vis=False, out_dir=output_dir) # out_dir 用于保存可视化结果（如果return_vis=True）
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
            person_prediction = frame_result['predictions'][0] # Assuming one list of predictions per frame
            if len(person_prediction) > 0 :
                keypoints_data = person_prediction[0] # Assuming one person
                keypoints = np.array(keypoints_data['keypoints']).tolist()
                keypoint_scores = np.array(keypoints_data['keypoint_scores']).tolist()
                
                # Combine keypoints with scores for simplicity in this example
                # In a real app, you might want to keep them separate or use scores for filtering
                combined_keypoints = []
                for kp, score in zip(keypoints, keypoint_scores):
                    combined_keypoints.append(kp + [score]) # [x, y, score]
                
                all_keypoints.append({
                    "frame_id": processed_frames,
                    "keypoints": combined_keypoints 
                })
            else: # No person detected in this frame by the model
                 all_keypoints.append({
                    "frame_id": processed_frames,
                    "keypoints": [] 
                })
        else: # No predictions for this frame
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
                "pose_data": [{"frame_id": 0, "keypoints": combined_keypoints}] # Treat as a single frame video
            }
    return {"image_info": {}, "pose_data": []}

```

**`app/analysis.py` (极其简化的对比示例)**

```python
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

    if not user_kps_frame or not target_kps_frame or not user_kps_frame["keypoints"] or not target_kps_frame["keypoints"]:
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
        feedback.append(f"Left Elbow Angle: Your: {user_le_angle:.1f}°, Target: {target_le_angle:.1f}°, Diff: {diff_le_angle:.1f}°")
        if abs(diff_le_angle) > 15: # Arbitrary threshold
             feedback.append(f"  Suggestion: Adjust your left elbow bend. {'Increase' if diff_le_angle < 0 else 'Decrease'} bending.")
    except Exception as e:
        feedback.append(f"Could not calculate Left Elbow Angle: {e}")

    # Example: Compare Left Knee Angle (hip-knee-ankle)
    # Keypoint indices: left_hip (11), left_knee (13), left_ankle (15)
    try:
        user_lk_angle = calculate_joint_angle(user_kps[11][:2], user_kps[13][:2], user_kps[15][:2])
        target_lk_angle = calculate_joint_angle(target_kps[11][:2], target_kps[13][:2], target_kps[15][:2])
        diff_lk_angle = user_lk_angle - target_lk_angle
        feedback.append(f"Left Knee Angle: Your: {user_lk_angle:.1f}°, Target: {target_lk_angle:.1f}°, Diff: {diff_lk_angle:.1f}°")
        if abs(diff_lk_angle) > 20: # Arbitrary threshold
            feedback.append(f"  Suggestion: Adjust your left knee bend. {'Increase' if diff_lk_angle < 0 else 'Decrease'} bending.")
    except Exception as e:
        feedback.append(f"Could not calculate Left Knee Angle: {e}")
        
    if not feedback:
        feedback.append("No specific issues found with these basic checks, or comparison failed.")

    return {"feedback": feedback}
```

**`app/utils.py`**
```python
import os
import uuid

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_uploaded_file(file, directory=UPLOAD_DIR):
    # Ensure filename is safe and unique
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(directory, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path, filename

```

**`app/main.py`**

```python
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import json
import shutil # For removing temp files

from .pose_estimator import extract_poses_from_video, extract_pose_from_image, get_inferencer
from .analysis import simple_compare_poses
from .utils import save_uploaded_file, UPLOAD_DIR, RESULTS_DIR

app = FastAPI()

# Mount static files (for HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads") # Serve uploaded videos

# In-memory storage for demo purposes. In a real app, use a DB or more persistent file storage.
TARGET_VIDEO_DATA = {} # Stores path and pose data for target
USER_VIDEO_DATA = {}   # Stores path and pose data for user video/image

@app.on_event("startup")
async def startup_event():
    # Pre-initialize the inferencer on startup to avoid delay on first request
    # This can take a while, so for development you might comment it out
    # and let it initialize on the first API call.
    print("Application startup: Initializing MMPose model...")
    try:
        get_inferencer()
        print("MMPose model initialized successfully.")
    except Exception as e:
        print(f"Error initializing MMPose model on startup: {e}")
        # You might want to raise an error or handle this gracefully
        # For now, we'll let it try again on the first API call if it fails here.


@app.post("/api/upload_target")
async def upload_target_video(video_file: UploadFile = File(...)):
    global TARGET_VIDEO_DATA
    try:
        file_path, filename = save_uploaded_file(video_file)
        pose_results = extract_poses_from_video(file_path) # This can be slow
        
        # Save pose data to a json file for persistence (optional for this demo)
        pose_json_path = os.path.join(RESULTS_DIR, f"target_{os.path.splitext(filename)[0]}.json")
        with open(pose_json_path, 'w') as f:
            json.dump(pose_results, f)
            
        TARGET_VIDEO_DATA = {"path": filename, "pose_data": pose_results, "type": "video"}
        return JSONResponse(content={
            "message": "Target video uploaded and processed.",
            "filename": filename,
            "pose_data_summary": f"{len(pose_results['pose_data'])} frames processed.",
            "video_url": f"/uploads/{filename}" # URL to access the video
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing target video: {str(e)}")

@app.post("/api/upload_user")
async def upload_user_media(media_file: UploadFile = File(...)):
    global USER_VIDEO_DATA
    try:
        file_path, filename = save_uploaded_file(media_file)
        
        content_type = media_file.content_type
        if content_type.startswith("video/"):
            pose_results = extract_poses_from_video(file_path)
            USER_VIDEO_DATA = {"path": filename, "pose_data": pose_results, "type": "video"}
            data_summary = f"{len(pose_results.get('pose_data',[]))} frames processed."
            media_url = f"/uploads/{filename}"
        elif content_type.startswith("image/"):
            pose_results = extract_pose_from_image(file_path)
            USER_VIDEO_DATA = {"path": filename, "pose_data": pose_results, "type": "image"}
            data_summary = "Image processed."
            media_url = f"/uploads/{filename}" # Images will also be served from uploads
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload video or image.")

        # Save pose data to a json file (optional)
        pose_json_path = os.path.join(RESULTS_DIR, f"user_{os.path.splitext(filename)[0]}.json")
        with open(pose_json_path, 'w') as f:
            json.dump(pose_results, f)

        return JSONResponse(content={
            "message": f"User {USER_VIDEO_DATA['type']} uploaded and processed.",
            "filename": filename,
            "pose_data_summary": data_summary,
            "media_url": media_url,
            "media_type": USER_VIDEO_DATA['type'],
            "raw_pose_data": pose_results # Send all pose data to frontend for drawing
        })
    except Exception as e:
        # Clean up uploaded file if processing fails
        if os.path.exists(file_path):
             os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing user media: {str(e)}")


@app.get("/api/compare")
async def compare_results(frame_id: int = 0): # Compare a specific frame, default to first
    if not TARGET_VIDEO_DATA or not USER_VIDEO_DATA:
        raise HTTPException(status_code=400, detail="Target or user video not uploaded/processed yet.")

    # For simplicity, we'll compare the specified frame_id.
    # A real app would need more sophisticated frame selection/alignment (e.g., DTW for sequences)
    
    # If user uploaded an image, it's always frame_id 0 for its pose_data
    user_frame_to_compare = 0 if USER_VIDEO_DATA.get("type") == "image" else frame_id
    
    # If target is an image, it's always frame_id 0
    target_frame_to_compare = 0 if TARGET_VIDEO_DATA.get("type") == "image" else frame_id


    # Ensure frame_id is valid for video data
    if USER_VIDEO_DATA.get("type") == "video" and frame_id >= len(USER_VIDEO_DATA["pose_data"]["pose_data"]):
        raise HTTPException(status_code=400, detail=f"User video frame_id {frame_id} out of bounds.")
    if TARGET_VIDEO_DATA.get("type") == "video" and frame_id >= len(TARGET_VIDEO_DATA["pose_data"]["pose_data"]):
         raise HTTPException(status_code=400, detail=f"Target video frame_id {frame_id} out of bounds.")


    comparison = simple_compare_poses(
        USER_VIDEO_DATA.get("pose_data"), 
        TARGET_VIDEO_DATA.get("pose_data"),
        frame_id=frame_id # This needs adjustment if one is image and other is video.
                          # For now, assumes if one is image, it is compared against frame 0 of video.
                          # Or if both are videos, against the same frame_id.
    )
    
    return JSONResponse(content={
        "comparison_feedback": comparison,
        "user_pose_frame_data": USER_VIDEO_DATA["pose_data"]["pose_data"][user_frame_to_compare] if USER_VIDEO_DATA.get("pose_data", {}).get("pose_data") else None,
        "target_pose_frame_data": TARGET_VIDEO_DATA["pose_data"]["pose_data"][target_frame_to_compare] if TARGET_VIDEO_DATA.get("pose_data", {}).get("pose_data") else None,
        "compared_frame_id": frame_id
    })

# Serve the main HTML page
from fastapi.responses import FileResponse
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    # This is for running with `python app/main.py`, but typically you'd run `uvicorn app.main:app --reload`
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**4. 前端代码 (`static/`)**

**`static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sport Video Analyzer</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Sport Video/Image Analyzer</h1>

    <div class="upload-section">
        <h2>1. Upload Target Athlete Media (Video or Image)</h2>
        <input type="file" id="targetFile" accept="video/*,image/*">
        <button onclick="uploadTarget()">Upload Target</button>
        <div id="targetUploadStatus"></div>
        <div id="targetMediaDisplay"></div>
    </div>

    <div class="upload-section">
        <h2>2. Upload Your Media (Video or Image)</h2>
        <input type="file" id="userFile" accept="video/*,image/*">
        <button onclick="uploadUser()">Upload Your Media</button>
        <div id="userUploadStatus"></div>
        <div id="userMediaDisplay"></div>
    </div>
    
    <div class="analysis-section">
        <h2>3. Analyze and Compare</h2>
        <label for="frameIdToCompare">Frame ID to Compare (for videos):</label>
        <input type="number" id="frameIdToCompare" value="0" min="0">
        <button onclick="getComparison()">Compare Poses</button>
        <div id="comparisonResult">
            <h3>Feedback:</h3>
            <pre id="feedbackText"></pre>
        </div>
    </div>

    <h2>Visual Comparison</h2>
    <div class="canvas-container">
        <div>
            <h3>Target Pose (Frame <span id="targetFrameDisp">0</span>)</h3>
            <canvas id="targetCanvas"></canvas>
        </div>
        <div>
            <h3>User Pose (Frame <span id="userFrameDisp">0</span>)</h3>
            <canvas id="userCanvas"></canvas>
        </div>
    </div>

    <script src="script.js"></script>
</body>
</html>
```

**`static/style.css`**

```css
body { font-family: sans-serif; margin: 20px; }
.upload-section, .analysis-section { margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }
input[type="file"], button { margin-top: 5px; margin-bottom: 10px; }
#targetUploadStatus, #userUploadStatus, #comparisonResult { margin-top: 10px; font-style: italic; }
video, img { max-width: 320px; max-height: 240px; border: 1px solid #ddd; }
.canvas-container { display: flex; gap: 20px; margin-top: 20px; }
canvas { border: 1px solid black; /* Dimensions will be set by JS */ }
pre { background-color: #f4f4f4; padding: 10px; border-radius: 3px; white-space: pre-wrap; }
```

**`static/script.js`**

```javascript
const targetFileUpload = document.getElementById('targetFile');
const userFileUpload = document.getElementById('userFile');
const targetUploadStatus = document.getElementById('targetUploadStatus');
const userUploadStatus = document.getElementById('userUploadStatus');
const targetMediaDisplay = document.getElementById('targetMediaDisplay');
const userMediaDisplay = document.getElementById('userMediaDisplay');

const feedbackText = document.getElementById('feedbackText');
const frameIdToCompareInput = document.getElementById('frameIdToCompare');

const targetCanvas = document.getElementById('targetCanvas');
const userCanvas = document.getElementById('userCanvas');
const targetCtx = targetCanvas.getContext('2d');
const userCtx = userCanvas.getContext('2d');

let currentTargetMedia = { url: null, type: null, poseData: null, videoInfo: null, imageInfo: null };
let currentUserMedia = { url: null, type: null, poseData: null, videoInfo: null, imageInfo: null };

// COCO keypoint connections (pairs of indices)
// Example: [L_Shoulder, R_Shoulder], [L_Shoulder, L_Elbow], ...
const COCO_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4],             // Head
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],    // Torso and Arms
    [5, 11], [6, 12], [11, 12],                 // Torso to Hips
    [11, 13], [13, 15], [12, 14], [14, 16]      // Legs
];
// Keypoint order for COCO (17 keypoints)
// 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
// 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
// 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
// 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

async function uploadMedia(fileInput, statusElement, displayElement, endpoint, mediaStore) {
    const file = fileInput.files[0];
    if (!file) {
        statusElement.textContent = 'Please select a file.';
        return;
    }

    statusElement.textContent = 'Uploading and processing...';
    displayElement.innerHTML = ''; // Clear previous media

    const formData = new FormData();
    if (endpoint.includes('target')) {
        formData.append('video_file', file); // Backend expects 'video_file' for target
    } else {
        formData.append('media_file', file); // Backend expects 'media_file' for user
    }


    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        statusElement.textContent = `${result.message} Summary: ${result.pose_data_summary}`;
        
        mediaStore.url = result.media_url || result.video_url; // Store the URL
        mediaStore.type = result.media_type || (file.type.startsWith('video/') ? 'video' : 'image');
        
        if (result.raw_pose_data) { // For user media, we get raw_pose_data
            mediaStore.poseData = result.raw_pose_data.pose_data;
            mediaStore.videoInfo = result.raw_pose_data.video_info;
            mediaStore.imageInfo = result.raw_pose_data.image_info;
        } else { // For target media, we don't send raw_pose_data back in this simplified version.
                 // We'd need another endpoint to fetch it, or include it in the response.
                 // For now, assume target pose data will be fetched during comparison.
            mediaStore.poseData = null; // Mark as needing fetch or not available for direct drawing yet
        }


        if (mediaStore.type === 'video') {
            const video = document.createElement('video');
            video.src = mediaStore.url;
            video.controls = true;
            video.width = 320; video.height = 240;
            displayElement.appendChild(video);
        } else if (mediaStore.type === 'image') {
            const img = document.createElement('img');
            img.src = mediaStore.url;
            img.width = 320; img.height = 240;
            displayElement.appendChild(img);
        }

    } catch (error) {
        console.error('Upload error:', error);
        statusElement.textContent = `Error: ${error.message}`;
    }
}

function uploadTarget() {
    uploadMedia(targetFileUpload, targetUploadStatus, targetMediaDisplay, '/api/upload_target', currentTargetMedia);
}

function uploadUser() {
    uploadMedia(userFileUpload, userUploadStatus, userMediaDisplay, '/api/upload_user', currentUserMedia);
}


async function getComparison() {
    if (!currentTargetMedia.url || !currentUserMedia.url) {
        feedbackText.textContent = 'Please upload both target and user media first.';
        return;
    }
    const frameId = parseInt(frameIdToCompareInput.value);
    if (isNaN(frameId) || frameId < 0) {
        feedbackText.textContent = 'Invalid Frame ID.';
        return;
    }

    feedbackText.textContent = 'Comparing...';
    document.getElementById('targetFrameDisp').textContent = frameId;
    document.getElementById('userFrameDisp').textContent = frameId;


    try {
        const response = await fetch(`/api/compare?frame_id=${frameId}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        
        if (result.comparison_feedback && result.comparison_feedback.feedback) {
            feedbackText.textContent = result.comparison_feedback.feedback.join('\n');
        } else if (result.comparison_feedback && result.comparison_feedback.error) {
            feedbackText.textContent = "Comparison Error: " + result.comparison_feedback.error;
        } else {
            feedbackText.textContent = 'Comparison done. No specific textual feedback generated by the simple comparison.';
        }
        
        // Draw poses
        // Target Pose
        if (result.target_pose_frame_data && result.target_pose_frame_data.keypoints) {
            const targetInfo = currentTargetMedia.type === 'video' ? currentTargetMedia.videoInfo : currentTargetMedia.imageInfo;
            if (targetInfo && targetInfo.width && targetInfo.height) {
                drawPoseOnCanvas(targetCtx, targetCanvas, result.target_pose_frame_data.keypoints, targetInfo.width, targetInfo.height, 'blue');
            } else {
                 console.warn("Target media info (width/height) not available for canvas scaling.");
                 drawPoseOnCanvas(targetCtx, targetCanvas, result.target_pose_frame_data.keypoints, 320, 240, 'blue'); // Fallback
            }
        } else {
            targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
            targetCtx.fillText("Target pose not available for this frame.", 10, 20);
        }

        // User Pose
        if (result.user_pose_frame_data && result.user_pose_frame_data.keypoints) {
             const userInfo = currentUserMedia.type === 'video' ? currentUserMedia.videoInfo : currentUserMedia.imageInfo;
             if (userInfo && userInfo.width && userInfo.height) {
                drawPoseOnCanvas(userCtx, userCanvas, result.user_pose_frame_data.keypoints, userInfo.width, userInfo.height, 'red');
             } else {
                 console.warn("User media info (width/height) not available for canvas scaling.");
                 drawPoseOnCanvas(userCtx, userCanvas, result.user_pose_frame_data.keypoints, 320, 240, 'red'); // Fallback
             }
        } else {
            userCtx.clearRect(0, 0, userCanvas.width, userCanvas.height);
            userCtx.fillText("User pose not available for this frame.", 10, 20);
        }

    } catch (error) {
        console.error('Comparison error:', error);
        feedbackText.textContent = `Error: ${error.message}`;
    }
}

function drawPoseOnCanvas(ctx, canvas, keypoints, originalWidth, originalHeight, color) {
    // Set canvas dimensions based on original aspect ratio, fitting into a max size
    const maxCanvasWidth = 320;
    const maxCanvasHeight = 240;
    let canvasWidth = originalWidth;
    let canvasHeight = originalHeight;

    if (canvasWidth > maxCanvasWidth) {
        const ratio = maxCanvasWidth / canvasWidth;
        canvasWidth = maxCanvasWidth;
        canvasHeight *= ratio;
    }
    if (canvasHeight > maxCanvasHeight) {
        const ratio = maxCanvasHeight / canvasHeight;
        canvasHeight = maxCanvasHeight;
        canvasWidth *= ratio;
    }
    
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#f0f0f0'; // Light gray background
    ctx.fillRect(0, 0, canvas.width, canvas.height);


    if (!keypoints || keypoints.length === 0) {
        ctx.fillStyle = 'black';
        ctx.fillText("No keypoints to draw.", 10, 20);
        return;
    }

    const scaleX = canvas.width / originalWidth;
    const scaleY = canvas.height / originalHeight;

    // Draw connections
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    COCO_CONNECTIONS.forEach(conn => {
        const p1_idx = conn[0];
        const p2_idx = conn[1];

        if (p1_idx < keypoints.length && p2_idx < keypoints.length) {
            const kp1 = keypoints[p1_idx]; // [x, y, score]
            const kp2 = keypoints[p2_idx];

            // Only draw if keypoints have a decent score (e.g., > 0.3)
            if (kp1[2] > 0.3 && kp2[2] > 0.3) {
                ctx.beginPath();
                ctx.moveTo(kp1[0] * scaleX, kp1[1] * scaleY);
                ctx.lineTo(kp2[0] * scaleX, kp2[1] * scaleY);
                ctx.stroke();
            }
        }
    });

    // Draw keypoints
    ctx.fillStyle = color;
    keypoints.forEach(kp => {
        if (kp[2] > 0.3) { // Only draw if score is decent
            ctx.beginPath();
            ctx.arc(kp[0] * scaleX, kp[1] * scaleY, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}

// Initial canvas setup
function setupInitialCanvas(ctx, canvas, text) {
    canvas.width = 320;
    canvas.height = 240;
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fillText(text, 10, 20);
}
setupInitialCanvas(targetCtx, targetCanvas, "Target pose will appear here.");
setupInitialCanvas(userCtx, userCanvas, "User pose will appear here.");

```

**5. 运行程序**

1.  确保你的 Conda 环境已激活 (`conda activate sport_analysis`).
2.  在项目根目录 (`sport_analyzer/`) 下运行 FastAPI 应用:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    `--reload` 会在代码更改时自动重启服务器，方便开发。
3.  打开浏览器访问 `http://localhost:8000` 或 `http://<你的IP地址>:8000`。

**重要说明和进一步改进方向**

*   **错误处理和鲁棒性**: 上述代码是原型，实际应用需要更健壮的错误处理。
*   **MMPose 模型下载**: MMPose Inferencer 首次使用时会自动下载模型。确保网络连接。你可以预先下载模型并指定本地路径。
*   **性能**: 视频姿态估计可能很慢。对于长视频，考虑：
    *   **采样帧**: 不是每一帧都分析。
    *   **异步处理**: 后端使用任务队列 (Celery, RQ) 处理耗时任务，前端轮询结果。
*   **对比算法**: `simple_compare_poses` 非常初级。实际需要：
    *   **DTW (Dynamic Time Warping)**: 用于对齐两个动作序列，即使它们速度不同。比较对齐后的姿态。
    *   **归一化**: 在比较前对姿态进行归一化（消除位置、大小、旋转差异），如 Procrustes 分析。
    *   **更复杂的规则或机器学习模型**: 用于评估和生成更精准的指导。
*   **3D 姿态估计**: 对于某些运动（如高尔夫挥杆），3D 姿态信息更有价值。MMPose 也支持一些 3D 模型。
*   **用户体验**:
    *   在视频上实时绘制骨骼和反馈。
    *   允许用户选择视频中的特定片段进行分析。
    *   更丰富的可视化。
*   **数据持久化**: `TARGET_VIDEO_DATA` 和 `USER_VIDEO_DATA` 是内存变量，服务重启会丢失。使用数据库或更可靠的文件存储。
*   **安全性**: 对上传文件进行严格校验。
*   **前端框架**: 对于更复杂的UI，使用 Vue.js 或 React 会使开发更高效。
*   **滑雪等特定运动**: 可能需要针对性的关键点定义或微调模型，因为标准 COCO 关键点可能不足以捕捉所有细节。

这个框架为你提供了一个起点。你的 4090 显卡对于 MMPose 的推理来说性能是足够的，关键在于设计好分析和对比的逻辑。祝你项目顺利！