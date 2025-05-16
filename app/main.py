from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import json
import traceback # 添加导入
import shutil  # For removing temp files

from .pose_estimator import extract_poses_from_video, extract_pose_from_image, get_inferencer
from .analysis import simple_compare_poses, compare_pose_sequences_dtw
from .utils import save_uploaded_file, UPLOAD_DIR, RESULTS_DIR
# from mmpose.apis import MMPoseInferencer # 直接导入

app = FastAPI()

# Mount static files (for HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")  # Serve uploaded videos

# In-memory storage for demo purposes. In a real app, use a DB or more persistent file storage.
TARGET_VIDEO_DATA = {}  # Stores path and pose data for target
USER_VIDEO_DATA = {}  # Stores path and pose data for user video/image


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


@app.post("/api/compare_videos")  # 改为 POST，因为可能需要发送整个视频的pose数据（或者后端从全局变量取）
async def compare_videos_endpoint():
    if not TARGET_VIDEO_DATA or not USER_VIDEO_DATA:
        raise HTTPException(status_code=400, detail="Target or user video not uploaded/processed yet.")

    # 确保是视频类型 (如果需要严格区分)
    if TARGET_VIDEO_DATA.get("type") != "video" or USER_VIDEO_DATA.get("type") != "video":
        raise HTTPException(status_code=400, detail="Video comparison requires both inputs to be videos.")

    target_poses = TARGET_VIDEO_DATA.get("pose_data", {}).get("pose_data", [])
    user_poses = USER_VIDEO_DATA.get("pose_data", {}).get("pose_data", [])

    if not target_poses or not user_poses:
        raise HTTPException(status_code=400, detail="Pose data missing for one or both videos.")

    comparison_result = compare_pose_sequences_dtw(target_poses, user_poses)

    if "error" in comparison_result:
        # 可以选择返回 200 OK 但包含错误信息，或返回 500/400
        return JSONResponse(status_code=400, content=comparison_result)

    return JSONResponse(content={
        "message": "Video comparison complete.",
        "dtw_results": comparison_result
    })


@app.post("/api/upload_target")
async def upload_target_video(video_file: UploadFile = File(...)): # 或者叫 media_file 更通用
    global TARGET_VIDEO_DATA
    try:
        file_path, filename = save_uploaded_file(video_file)
        content_type = video_file.content_type # 获取 content_type

        # 根据 content_type 处理视频或图像
        if content_type.startswith("video/"):
            pose_results = extract_poses_from_video(file_path)
            TARGET_VIDEO_DATA = {"path": filename, "pose_data": pose_results, "type": "video"}
            media_type = "video"
            media_url = f"/uploads/{filename}"
            # 提取 video_info 以便返回给前端
            video_info_to_return = pose_results.get("video_info")
            image_info_to_return = None

        elif content_type.startswith("image/"):
            pose_results = extract_pose_from_image(file_path)
            TARGET_VIDEO_DATA = {"path": filename, "pose_data": pose_results, "type": "image"}
            media_type = "image"
            media_url = f"/uploads/{filename}"
            # 提取 image_info 以便返回给前端
            image_info_to_return = pose_results.get("image_info")
            video_info_to_return = None
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for target. Please upload video or image.")


        # pose_json_path = os.path.join(RESULTS_DIR, f"target_{os.path.splitext(filename)[0]}.json")
        # with open(pose_json_path, 'w') as f:
        #     json.dump(pose_results, f)

        return JSONResponse(content={
            "message": f"Target {media_type} uploaded and processed.",
            "filename": filename,
            # 根据你的前端，pose_data_summary 可能需要调整
            "pose_data_summary": f"{len(pose_results.get('pose_data', []))} frames/objects processed.",
            "media_url": media_url, # 统一键名
            "media_type": media_type, # 添加 media_type
            # 返回原始的 video_info 或 image_info
            "raw_pose_data": { # 保持与 user 端一致的结构，方便前端处理
                "video_info": video_info_to_return,
                "image_info": image_info_to_return,
                # 注意：这里不直接返回 pose_data, 因为 target 的 pose_data 是通过 /api/compare 获取的
                # 如果你希望 target 上传后就能直接画单帧，可以考虑返回第一帧的 pose
            }
        })
    except Exception as e:
        print("!!! ERROR IN /api/upload_target !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing target media: {str(e)}")


@app.post("/api/upload_user")
async def upload_user_media(media_file: UploadFile = File(...)):
    global USER_VIDEO_DATA
    try:
        file_path, filename = save_uploaded_file(media_file)

        content_type = media_file.content_type
        if content_type.startswith("video/"):
            pose_results = extract_poses_from_video(file_path)
            USER_VIDEO_DATA = {"path": filename, "pose_data": pose_results, "type": "video"}
            data_summary = f"{len(pose_results.get('pose_data', []))} frames processed."
            media_url = f"/uploads/{filename}"
        elif content_type.startswith("image/"):
            pose_results = extract_pose_from_image(file_path)
            USER_VIDEO_DATA = {"path": filename, "pose_data": pose_results, "type": "image"}
            data_summary = "Image processed."
            media_url = f"/uploads/{filename}"  # Images will also be served from uploads
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
            "raw_pose_data": pose_results  # Send all pose data to frontend for drawing
        })
    except Exception as e:
        # Clean up uploaded file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing user media: {str(e)}")


@app.get("/api/compare")
async def compare_results(frame_id: int = 0):  # Compare a specific frame, default to first
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
        frame_id=frame_id  # This needs adjustment if one is image and other is video.
        # For now, assumes if one is image, it is compared against frame 0 of video.
        # Or if both are videos, against the same frame_id.
    )

    return JSONResponse(content={
        "comparison_feedback": comparison,
        "user_pose_frame_data": USER_VIDEO_DATA["pose_data"]["pose_data"][user_frame_to_compare] if USER_VIDEO_DATA.get(
            "pose_data", {}).get("pose_data") else None,
        "target_pose_frame_data": TARGET_VIDEO_DATA["pose_data"]["pose_data"][
            target_frame_to_compare] if TARGET_VIDEO_DATA.get("pose_data", {}).get("pose_data") else None,
        "compared_frame_id": frame_id
    })


# 在 app/main.py 文件中
@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutdown: Cleaning up resources...")
    # 在这里添加任何自定义清理代码
    # 注意：MMPoseInferencer 应该会管理它自己的资源。


# Serve the main HTML page
from fastapi.responses import FileResponse


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


if __name__ == "__main__":
    import uvicorn

    # This is for running with `python app/main.py`, but typically you'd run `uvicorn app.main:app --reload`
    uvicorn.run(app, host="0.0.0.0", port=8000)