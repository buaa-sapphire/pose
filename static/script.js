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


// 新增变量用于存储DTW对齐结果和当前查看的对齐帧索引
let dtwAlignmentData = null;
let currentAlignedPairIndex = 0;
let dtwTargetCtx = null;
let dtwUserCtx = null;


// 在 setupInitialCanvas 或页面加载时初始化新的 canvas
function initializeDTWCanvases() {
    const dtwTargetCanvas = document.getElementById('dtwTargetCanvas');
    const dtwUserCanvas = document.getElementById('dtwUserCanvas');
    if (dtwTargetCanvas && dtwUserCanvas) {
        dtwTargetCtx = dtwTargetCanvas.getContext('2d');
        dtwUserCtx = dtwUserCanvas.getContext('2d');
        setupInitialCanvas(dtwTargetCtx, dtwTargetCanvas, "Aligned target pose");
        setupInitialCanvas(dtwUserCtx, dtwUserCanvas, "Aligned user pose");
    }
}
// 调用初始化
window.addEventListener('load', initializeDTWCanvases);


async function getVideosComparison() {
    if (!currentTargetMedia.url || !currentUserMedia.url) { // 简单检查，后端也会检查
        feedbackText.textContent = 'Please upload both target and user videos first.';
        return;
    }
     if (currentTargetMedia.type !== 'video' || currentUserMedia.type !== 'video') {
        feedbackText.textContent = 'Full video comparison is only for video types.';
        return;
    }

    feedbackText.textContent = 'Comparing full videos (this may take a moment)...';
    document.getElementById('videoOverallSimilarity').textContent = '';
    document.getElementById('videoMostDifferentFramesFeedback').innerHTML = ''; // 使用 innerHTML 来插入多行
    document.querySelector('.aligned-frames-viewer').style.display = 'none'; // 隐藏帧查看器
    dtwAlignmentData = null; // 重置

    try {
        const response = await fetch('/api/compare_videos', { method: 'POST' });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || errorData.error || `HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        console.log("Full Video Comparison Result:", result); // 调试用

        if (result.dtw_results && result.dtw_results.normalized_dtw_distance !== undefined) {
            dtwAlignmentData = result.dtw_results; // 存储完整的对齐数据

            let overallFeedback = `Overall Video Similarity (Normalized DTW Distance): ${dtwAlignmentData.normalized_dtw_distance.toFixed(4)}\n`;
            overallFeedback += `(Lower distance means more similar sequences)\n`;
            // ... (可以添加定性评价) ...
            document.getElementById('videoOverallSimilarity').textContent = overallFeedback;

            let diffFramesHtml = "<p>Details for most different aligned frames:</p><ul>";
            dtwAlignmentData.most_different_frames_feedback.forEach(item => {
                diffFramesHtml += `<li>Target Frame ${item.target_frame_id} vs User Frame ${item.user_frame_id} (Diff Score: ${item.difference_score.toFixed(2)}):<ul>`;
                item.detailed_feedback.forEach(fb_line => {
                    diffFramesHtml += `<li>${fb_line.replace(/^---.*---/, '').trim()}</li>`; // 清理一下标题
                });
                diffFramesHtml += `</ul></li>`;
            });
            diffFramesHtml += "</ul>";
            document.getElementById('videoMostDifferentFramesFeedback').innerHTML = diffFramesHtml;

            feedbackText.textContent = "Full video comparison done. See results below.";

            // 如果有对齐路径，显示帧查看器
            if (dtwAlignmentData.original_target_frame_ids_in_path && dtwAlignmentData.original_target_frame_ids_in_path.length > 0) {
                document.querySelector('.aligned-frames-viewer').style.display = 'block';
                currentAlignedPairIndex = 0;
                displayCurrentAlignedPair();
            }

        } else if (result.dtw_results && result.dtw_results.error) {
            document.getElementById('videoOverallSimilarity').textContent = `Video Comparison Error: ${result.dtw_results.error}`;
        } else {
            document.getElementById('videoOverallSimilarity').textContent = 'Could not get DTW results for video comparison.';
        }


    } catch (error) {
        console.error('Video comparison error:', error);
        document.getElementById('videoComparisonFeedback').textContent = `Error: ${error.message}`;
         feedbackText.textContent = "Error during full video comparison.";
    }
}

function displayCurrentAlignedPair() {
    // 1. 检查DTW对齐数据是否存在且有效
    if (!dtwAlignmentData ||
        !dtwAlignmentData.original_target_frame_ids_in_path ||
        dtwAlignmentData.original_target_frame_ids_in_path.length === 0 ||
        !dtwAlignmentData.original_user_frame_ids_in_path || // 也检查user的路径
        dtwAlignmentData.original_user_frame_ids_in_path.length !== dtwAlignmentData.original_target_frame_ids_in_path.length // 确保路径长度一致
    ) {
        console.warn("displayCurrentAlignedPair: DTW alignment data is not ready, incomplete, or paths have different lengths.");
        // 可以考虑在这里清除或重置相关的显示区域
        document.getElementById('alignedFrameIndicator').textContent = "N/A";
        document.getElementById('dtwTargetFrameId').textContent = "?";
        document.getElementById('dtwUserFrameId').textContent = "?";
        if (dtwTargetCtx) setupInitialCanvas(dtwTargetCtx, document.getElementById('dtwTargetCanvas'), "DTW data unavailable.");
        if (dtwUserCtx) setupInitialCanvas(dtwUserCtx, document.getElementById('dtwUserCanvas'), "DTW data unavailable.");
        document.getElementById('currentAlignedPairFeedback').textContent = "DTW alignment data not available to display pair.";
        return;
    }

    const pathLen = dtwAlignmentData.original_target_frame_ids_in_path.length;

    // 2. 检查当前对齐帧索引是否在有效范围内
    if (currentAlignedPairIndex < 0 || currentAlignedPairIndex >= pathLen) {
        console.warn(`displayCurrentAlignedPair: currentAlignedPairIndex (${currentAlignedPairIndex}) is out of bounds (0-${pathLen - 1}).`);
        // 可以选择重置索引或不执行任何操作
        // currentAlignedPairIndex = Math.max(0, Math.min(currentAlignedPairIndex, pathLen - 1)); // Clamp index
        return;
    }

    // 3. 获取当前对齐帧的Target和User的原始帧ID
    const targetFrameIdFromDTW = dtwAlignmentData.original_target_frame_ids_in_path[currentAlignedPairIndex];
    const userFrameIdFromDTW = dtwAlignmentData.original_user_frame_ids_in_path[currentAlignedPairIndex];

    // 更新UI显示当前的帧ID和对齐进度
    document.getElementById('alignedFrameIndicator').textContent = `Pair ${currentAlignedPairIndex + 1}/${pathLen}`;
    document.getElementById('dtwTargetFrameId').textContent = targetFrameIdFromDTW;
    document.getElementById('dtwUserFrameId').textContent = userFrameIdFromDTW;

    // --- 开始详细的日志记录 ---
    console.log(`--- Displaying Aligned Pair - Index: ${currentAlignedPairIndex} ---`);
    console.log(`  DTW Path Target Frame ID: ${targetFrameIdFromDTW}`);
    console.log(`  DTW Path User Frame ID: ${userFrameIdFromDTW}`);

    // 4. 准备 Target Pose 的数据
    // console.log("  currentTargetMedia (for Target):", JSON.parse(JSON.stringify(currentTargetMedia))); // 深拷贝打印，避免引用问题
    const targetKps = getKeypointsForFrameId(currentTargetMedia.poseData, targetFrameIdFromDTW);
    const targetInfo = currentTargetMedia.type === 'video' ? currentTargetMedia.videoInfo : currentTargetMedia.imageInfo;
    // console.log(`  Target Keypoints for Frame ${targetFrameIdFromDTW}:`, targetKps);
    // console.log(`  Target Media Info:`, targetInfo);

    // 5. 准备 User Pose 的数据
    // console.log("  currentUserMedia (for User):", JSON.parse(JSON.stringify(currentUserMedia))); // 深拷贝打印
    const userKps = getKeypointsForFrameId(currentUserMedia.poseData, userFrameIdFromDTW);
    const userInfo = currentUserMedia.type === 'video' ? currentUserMedia.videoInfo : currentUserMedia.imageInfo;
    // console.log(`  User Keypoints for Frame ${userFrameIdFromDTW}:`, userKps);
    // console.log(`  User Media Info:`, userInfo);

    // 6. 获取Canvas上下文 (确保它们已在页面加载时初始化到 dtwTargetCtx 和 dtwUserCtx)
    const targetCanvasEl = document.getElementById('dtwTargetCanvas');
    const userCanvasEl = document.getElementById('dtwUserCanvas');
    // dtwTargetCtx 和 dtwUserCtx 应该已经是全局或可访问的上下文变量

    // 7. 绘制 Target Pose
    console.log("Checking variables for pose analysis:");
    console.log("targetKps:", targetKps);
    console.log("targetInfo:", targetInfo);

    if (!targetKps) {
        console.error("Error: targetKps is undefined or null.");
    }
    if (!targetInfo) {
        console.error("Error: targetInfo is undefined or null.");
    } else {
        console.log("targetInfo.width:", targetInfo.width);
        console.log("targetInfo.height:", targetInfo.height);

        if (!targetInfo.width || !targetInfo.height) {
            console.error("Error: targetInfo.width or targetInfo.height is missing or invalid.");
        }
    }

    if (targetKps && targetInfo && targetInfo.width && targetInfo.height) {
        console.log(`  Drawing Target Pose: Frame ${targetFrameIdFromDTW}, Original W/H ${targetInfo.width}/${targetInfo.height}`);
        drawPoseOnCanvas(dtwTargetCtx, targetCanvasEl, targetKps, targetInfo.width, targetInfo.height, 'cyan');
    } else {
        console.error(`  Failed to draw Target Pose. Reason(s):`);
        if (!targetKps) {
            console.error(`    - targetKps is null/undefined for targetFrameId: ${targetFrameIdFromDTW}.`);
            if (currentTargetMedia.poseData && Array.isArray(currentTargetMedia.poseData)) {
                console.log("    Available frame_ids in currentTargetMedia.poseData:", currentTargetMedia.poseData.map(f => f ? f.frame_id : 'null_frame_in_array'));
            } else {
                console.log("    currentTargetMedia.poseData itself is not an array or is null/undefined:", currentTargetMedia.poseData);
            }
        }
        if (!targetInfo) {
            console.error(`    - targetInfo is null or undefined for Target.`);
        } else if (!targetInfo.width || !targetInfo.height) {
            console.error(`    - targetInfo.width or targetInfo.height is missing/invalid for Target. targetInfo:`, targetInfo);
        }
        setupInitialCanvas(dtwTargetCtx, targetCanvasEl, "Target pose data missing for this aligned frame.");
    }

    // 8. 绘制 User Pose
    if (userKps && userInfo && userInfo.width && userInfo.height) {
        // console.log(`  Drawing User Pose: Frame ${userFrameIdFromDTW}, Original W/H ${userInfo.width}/${userInfo.height}`);
        drawPoseOnCanvas(dtwUserCtx, userCanvasEl, userKps, userInfo.width, userInfo.height, 'magenta');
    } else {
        console.error(`  Failed to draw User Pose. Reason(s):`);
        if (!userKps) {
            console.error(`    - userKps is null or undefined for userFrameId: ${userFrameIdFromDTW}.`);
             if (currentUserMedia.poseData && Array.isArray(currentUserMedia.poseData)) {
                console.log("    Available frame_ids in currentUserMedia.poseData:", currentUserMedia.poseData.map(f => f ? f.frame_id : 'null_frame_in_array'));
            } else {
                console.log("    currentUserMedia.poseData itself is not an array or is null/undefined:", currentUserMedia.poseData);
            }
        }
        if (!userInfo) {
            console.error(`    - userInfo is null or undefined for User.`);
        } else if (!userInfo.width || !userInfo.height) {
            console.error(`    - userInfo.width or userInfo.height is missing/invalid for User. userInfo:`, userInfo);
        }
        setupInitialCanvas(dtwUserCtx, userCanvasEl, "User pose data missing for this aligned frame.");
    }

    // 9. 显示当前对齐帧的详细对比反馈
    let currentPairFbText = `Feedback for Aligned Pair (Target Frame ${targetFrameIdFromDTW} vs User Frame ${userFrameIdFromDTW}):\n`;
    const foundFbItem = dtwAlignmentData.most_different_frames_feedback.find(
        item => item.target_frame_id === targetFrameIdFromDTW && item.user_frame_id === userFrameIdFromDTW
    );

    if (foundFbItem && foundFbItem.detailed_feedback) {
        // 清理和格式化反馈
        currentPairFbText += foundFbItem.detailed_feedback
            .map(line => line.replace(/^---.*---/, '').trim()) // 移除标题行并修剪空格
            .filter(line => line.length > 0) // 移除空行
            .join('\n');
    } else {
        currentPairFbText += "\n(This specific pair is not among the top N most different frames for detailed feedback, or feedback is unavailable.)";
        // 考虑：如果后端没有返回所有对齐帧的 detailed_feedback，
        // 这里可以调用一个轻量级的前端单帧比较函数（如果需要实时反馈），
        // 或者提示用户此帧的详细反馈未预先计算。
        // e.g., if (typeof frontendSimpleCompare === 'function') {
        //    const feComparison = frontendSimpleCompare(targetKps, userKps); // 需要一个这样的函数
        //    currentPairFbText += "\n\nFrontend Basic Check:\n" + feComparison.join('\n');
        // }
    }
    document.getElementById('currentAlignedPairFeedback').textContent = currentPairFbText;
    console.log(`--- End Displaying Aligned Pair ---`);
}

//function displayCurrentAlignedPair() {
//    if (!dtwAlignmentData || !dtwAlignmentData.original_target_frame_ids_in_path || dtwAlignmentData.original_target_frame_ids_in_path.length === 0) {
//        return;
//    }
//    const pathLen = dtwAlignmentData.original_target_frame_ids_in_path.length;
//    if (currentAlignedPairIndex < 0 || currentAlignedPairIndex >= pathLen) {
//        return;
//    }
//
//    const targetFrameId = dtwAlignmentData.original_target_frame_ids_in_path[currentAlignedPairIndex];
//    const userFrameId = dtwAlignmentData.original_user_frame_ids_in_path[currentAlignedPairIndex];
//
//    document.getElementById('alignedFrameIndicator').textContent = `Pair ${currentAlignedPairIndex + 1}/${pathLen}`;
//    document.getElementById('dtwTargetFrameId').textContent = targetFrameId;
//    document.getElementById('dtwUserFrameId').textContent = userFrameId;
//
//    // 获取对应帧的keypoints
//    const targetKps = getKeypointsForFrameId(currentTargetMedia.poseData, targetFrameId); // poseData是帧列表
//    const userKps = getKeypointsForFrameId(currentUserMedia.poseData, userFrameId);     // poseData是帧列表
//
//    const targetInfo = currentTargetMedia.type === 'video' ? currentTargetMedia.videoInfo : currentTargetMedia.imageInfo;
//    const userInfo = currentUserMedia.type === 'video' ? currentUserMedia.videoInfo : currentUserMedia.imageInfo;
//
//    if (targetKps && targetInfo && targetInfo.width && targetInfo.height) {
//        drawPoseOnCanvas(dtwTargetCtx, document.getElementById('dtwTargetCanvas'), targetKps, targetInfo.width, targetInfo.height, 'cyan');
//    } else {
//        setupInitialCanvas(dtwTargetCtx, document.getElementById('dtwTargetCanvas'), "Target pose data missing for this aligned frame.");
//    }
//
//    if (userKps && userInfo && userInfo.width && userInfo.height) {
//        drawPoseOnCanvas(dtwUserCtx, document.getElementById('dtwUserCanvas'), userKps, userInfo.width, userInfo.height, 'magenta');
//    } else {
//         setupInitialCanvas(dtwUserCtx, document.getElementById('dtwUserCanvas'), "User pose data missing for this aligned frame.");
//    }
//
//    // 显示当前对齐帧的详细对比 (需要从后端获取或在前端重新计算)
//    // 为了简单，我们先从 dtwAlignmentData.most_different_frames_feedback 中查找，如果存在的话
//    let currentPairFbText = `Comparing Target Frame ${targetFrameId} with User Frame ${userFrameId}\n`;
//    const foundFb = dtwAlignmentData.most_different_frames_feedback.find(
//        item => item.target_frame_id === targetFrameId && item.user_frame_id === userFrameId
//    );
//    if (foundFb) {
//        currentPairFbText += foundFb.detailed_feedback.join('\n');
//    } else {
//        // 如果不在 top N 差异中，可以提示用户或调用一个轻量级的单帧比较
//        // 或者后端应该返回所有对齐帧的差异（如果性能允许）
//         currentPairFbText += "(This pair is not in the top N most different frames. Detailed feedback might be limited here or recalculate if needed)";
//         // 你可以调用一个简化的前端比较函数，或者从后端获取更详细的
//    }
//    document.getElementById('currentAlignedPairFeedback').textContent = currentPairFbText;
//}

function getKeypointsForFrameId(poseDataArray, frameId) {
    if (!poseDataArray) return null;
    const frame = poseDataArray.find(f => f.frame_id === frameId);
    return frame ? frame.keypoints : null;
}

function prevAlignedFrame() {
    if (dtwAlignmentData && currentAlignedPairIndex > 0) {
        currentAlignedPairIndex--;
        displayCurrentAlignedPair();
    }
}

function nextAlignedFrame() {
    if (dtwAlignmentData && currentAlignedPairIndex < dtwAlignmentData.original_target_frame_ids_in_path.length - 1) {
        currentAlignedPairIndex++;
        displayCurrentAlignedPair();
    }
}


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

            console.log('Target Pose Keypoints:', result.target_pose_frame_data.keypoints);

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


