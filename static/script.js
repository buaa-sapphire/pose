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