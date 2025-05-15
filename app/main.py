from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks
import os
import json
import logging

import shutil  # For removing temp files

from .pose_estimator import extract_poses_from_video, extract_pose_from_image, get_inferencer
from .analysis import simple_compare_poses
from .utils import save_uploaded_file, UPLOAD_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files (for HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")  # Serve uploaded videos

# In-memory storage for demo purposes. In a real app, use a DB or more persistent file storage.
TARGET_VIDEO_DATA = {}  # Stores path and pose data for target
USER_VIDEO_LIST = []  # 存储所有用户上传的视频


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
async def upload_target_video(video_file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    print(f"Received file: {video_file.filename}")  # 打印接收到的文件名
    global TARGET_VIDEO_DATA
    try:
        # Save the uploaded file
        file_path, filename = save_uploaded_file(video_file)

        # Simulate progress updates (if needed)
        def process_in_background():
            pose_results = extract_poses_from_video(file_path)  # This can be slow
            pose_json_path = os.path.join(RESULTS_DIR, f"target_{os.path.splitext(filename)[0]}.json")
            with open(pose_json_path, 'w') as f:
                json.dump(pose_results, f)
            TARGET_VIDEO_DATA = {"path": filename, "pose_data": pose_results, "type": "video"}

        # Run processing in the background
        background_tasks.add_task(process_in_background)

        return JSONResponse(content={
            "message": "Target video is being processed.",
            "filename": filename,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing target video: {str(e)}")


@app.post("/api/upload_user")
async def upload_user_media(media_file: UploadFile = File(...)):
    try:
        # Log received file information
        logger.info(f"Received file: {media_file.filename}, Content-Type: {media_file.content_type}")

        # Save the uploaded file
        file_path, filename = save_uploaded_file(media_file)
        logger.info(f"File saved to: {file_path}")

        content_type = media_file.content_type
        if content_type.startswith("video/"):
            pose_results = extract_poses_from_video(file_path)
            video_data = {"path": filename, "pose_data": pose_results, "type": "video"}
            USER_VIDEO_LIST.append(video_data)  # 将新上传的视频添加到列表中
            data_summary = f"{len(pose_results.get('pose_data', []))} frames processed."
            media_url = f"/uploads/{filename}"
        elif content_type.startswith("image/"):
            pose_results = extract_pose_from_image(file_path)
            video_data = {"path": filename, "pose_data": pose_results, "type": "image"}
            USER_VIDEO_LIST.append(video_data)  # 将新上传的图像添加到列表中
            data_summary = "Image processed."
            media_url = f"/uploads/{filename}"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload video or image.")

        # Save pose data to a json file
        pose_json_path = os.path.join(RESULTS_DIR, f"user_{os.path.splitext(filename)[0]}.json")
        with open(pose_json_path, 'w') as f:
            json.dump(pose_results, f)

        return JSONResponse(content={
            "message": f"User {video_data['type']} uploaded and processed.",
            "filename": filename,
            "pose_data_summary": data_summary,
            "media_url": media_url,
            "media_type": video_data['type'],
            "raw_pose_data": pose_results
        })
    except Exception as e:
        logger.error(f"Error processing user media: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing user media: {str(e)}")


@app.get("/api/compare")
async def compare_results(frame_id: int = 0):  # Compare a specific frame, default to first
    if not TARGET_VIDEO_DATA or not USER_VIDEO_LIST:
        raise HTTPException(status_code=400, detail="Target or user video not uploaded/processed yet.")

    # For simplicity, we'll compare the specified frame_id.
    # A real app would need more sophisticated frame selection/alignment (e.g., DTW for sequences)

    # If user uploaded an image, it's always frame_id 0 for its pose_data
    user_frame_to_compare = 0 if USER_VIDEO_LIST[-1].get("type") == "image" else frame_id

    # If target is an image, it's always frame_id 0
    target_frame_to_compare = 0 if TARGET_VIDEO_DATA.get("type") == "image" else frame_id

    # Ensure frame_id is valid for video data
    if USER_VIDEO_LIST[-1].get("type") == "video" and frame_id >= len(USER_VIDEO_LIST[-1]["pose_data"]["pose_data"]):
        raise HTTPException(status_code=400, detail=f"User video frame_id {frame_id} out of bounds.")
    if TARGET_VIDEO_DATA.get("type") == "video" and frame_id >= len(TARGET_VIDEO_DATA["pose_data"]["pose_data"]):
        raise HTTPException(status_code=400, detail=f"Target video frame_id {frame_id} out of bounds.")

    comparison = simple_compare_poses(
        USER_VIDEO_LIST[-1].get("pose_data"),
        TARGET_VIDEO_DATA.get("pose_data"),
        frame_id=frame_id  # This needs adjustment if one is image and other is video.
        # For now, assumes if one is image, it is compared against frame 0 of video.
        # Or if both are videos, against the same frame_id.
    )

    return JSONResponse(content={
        "comparison_feedback": comparison,
        "user_pose_frame_data": USER_VIDEO_LIST[-1]["pose_data"]["pose_data"][user_frame_to_compare] if USER_VIDEO_LIST[-1].get(
            "pose_data", {}).get("pose_data") else None,
        "target_pose_frame_data": TARGET_VIDEO_DATA["pose_data"]["pose_data"][
            target_frame_to_compare] if TARGET_VIDEO_DATA.get("pose_data", {}).get("pose_data") else None,
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