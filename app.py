# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import torch
import copy
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# 导入detect_rec_plate中的核心函数
from detect_rec_plate import (
    load_model, init_model, det_rec_plate, draw_result,
    process_video, device as drp_device, get_best_plate_frame
)

# 初始化Flask应用
app = Flask(__name__, static_folder='.')

# 配置项
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
FRAME_FOLDER = 'frames'  # 新增：存储视频关键帧
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB

# 创建必要文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)  # 创建关键帧存储文件夹

# 设备配置
device = drp_device

# 模型路径配置
DETECT_MODEL_PATH = os.path.join('weights', 'yolov8s.pt')
REC_MODEL_PATH = os.path.join('weights', 'plate_rec_color.pth')

# 全局加载模型（全部改为英文，彻底避免乱码）
print("Loading license plate detection and recognition models...")
try:
    detect_model = load_model(DETECT_MODEL_PATH, device)
    plate_rec_model = init_model(device, REC_MODEL_PATH, is_color=True)
    detect_model.eval()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise


# 辅助函数：检查文件扩展名是否合法
def allowed_file(filename, file_type):
    if file_type == 'image':
        allowed = ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        allowed = ALLOWED_VIDEO_EXTENSIONS
    else:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


# 主页路由
@app.route('/')
def index():
    """Return frontend page"""
    try:
        return send_file('index.html')
    except Exception as e:
        return f"index.html not found: {str(e)}", 404


# 上传识别接口（支持图片/视频）
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image/video upload and return recognition result"""
    try:
        # 检查文件是否存在
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected', 'success': False}), 400

        file = request.files['file']
        file_type = request.form.get('file_type', 'image')

        if file.filename == '':
            return jsonify({'error': 'Empty filename', 'success': False}), 400

        # 校验文件类型和大小
        if not allowed_file(file.filename, file_type):
            allowed = ALLOWED_IMAGE_EXTENSIONS if file_type == 'image' else ALLOWED_VIDEO_EXTENSIONS
            return jsonify({
                'error': f'Unsupported file format. Supported: {",".join(allowed)}',
                'success': False
            }), 400

        # 校验文件大小
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_type == 'image' and file_size > MAX_IMAGE_SIZE:
            return jsonify({'error': 'Image size exceeds 5MB limit', 'success': False}), 400
        elif file_type == 'video' and file_size > MAX_VIDEO_SIZE:
            return jsonify({'error': 'Video size exceeds 100MB limit', 'success': False}), 400

        # 保存上传文件
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        start_time = time.time()
        result_filename = f"result_{int(time.time())}_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        plate_list = []
        frame_url = ""  # 视频关键帧URL

        # 处理图片
        if file_type == 'image':
            img = cv2.imread(upload_path)
            if img is None:
                return jsonify({'error': 'Cannot read image file', 'success': False}), 400

            img_ori = copy.deepcopy(img)
            result_list = det_rec_plate(img, img_ori, detect_model, plate_rec_model)
            result_img, result_str = draw_result(img_ori, result_list)
            cv2.imwrite(result_path, result_img)

            # 提取车牌
            plate_list = [res['plate_no'] for res in result_list if res['plate_no']]
            if not plate_list:
                plate_list = ['未识别到车牌']

        # 处理视频
        elif file_type == 'video':
            try:
                # 1. 获取最优关键帧（带框选）+ 最优车牌列表（仅保留关键帧结果）
                best_frame, plate_list, _ = get_best_plate_frame(upload_path, detect_model, plate_rec_model)

                # 保存关键帧
                if best_frame is not None:
                    frame_filename = f"frame_{int(time.time())}_{os.path.splitext(filename)[0]}.jpg"
                    frame_path = os.path.join(FRAME_FOLDER, frame_filename)
                    cv2.imwrite(frame_path, best_frame)
                    frame_url = f'/frames/{frame_filename}'
                else:
                    frame_url = ""

                # 2. 生成带标注的视频（仅处理视频，不获取其车牌结果）
                video_save_path = process_video(
                    upload_path, detect_model, plate_rec_model, result_path, frame_interval=2
                )

                # 兜底：如果关键帧未识别到车牌，显示未识别
                if not plate_list or plate_list == ['']:
                    plate_list = ['未识别到车牌']

            except Exception as e:
                return jsonify({'error': f'Video processing failed: {str(e)}', 'success': False}), 500

        # 计算处理时间
        process_time = round(time.time() - start_time, 2)

        # 返回结果（新增frame_url字段）
        return jsonify({
            'file_url': f'/results/{result_filename}',
            'frame_url': frame_url,  # 视频关键帧URL（图片识别为空）
            'plate': plate_list,
            'success': True,
            'process_time': process_time,
            'file_type': file_type
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}', 'success': False}), 500


# 提供结果文件访问（图片/视频）
@app.route('/results/<filename>')
def send_result(filename):
    """Return recognition result file (image/video)"""
    try:
        file_path = os.path.join(RESULT_FOLDER, filename)
        # 根据扩展名判断文件类型
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in ALLOWED_VIDEO_EXTENSIONS:
            return send_file(file_path, mimetype=f'video/{ext}')
        else:
            return send_file(file_path)
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}', 'success': False}), 404


# 新增：提供关键帧访问
@app.route('/frames/<filename>')
def send_frame(filename):
    """Return video key frame"""
    try:
        file_path = os.path.join(FRAME_FOLDER, filename)
        return send_file(file_path)
    except Exception as e:
        return jsonify({'error': f'Frame not found: {str(e)}', 'success': False}), 404


# 主函数
if __name__ == '__main__':
    # 配置文件大小限制
    app.config['MAX_CONTENT_LENGTH'] = MAX_VIDEO_SIZE  # 最大支持100MB

    # 检查关键文件（英文提示）
    if not os.path.exists('index.html'):
        print("Error: index.html not found in root directory!")
        exit(1)
    if not os.path.exists(DETECT_MODEL_PATH):
        print(f"Error: Detection model not found: {DETECT_MODEL_PATH}")
        exit(1)
    if not os.path.exists(REC_MODEL_PATH):
        print(f"Error: Recognition model not found: {REC_MODEL_PATH}")
        exit(1)

    # 英文启动提示
    print("License plate recognition system starting...")
    print("Support: Image (PNG/JPG/JPEG) & Video (MP4/AVI/MOV)")
    print("Access URL: http://localhost:5000")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )