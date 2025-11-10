import os
import time
import threading
import logging
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify, send_file
from ultralytics import YOLO
from datetime import datetime

# 初始化日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 初始化Flask应用
app = Flask(__name__)

# 配置参数
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'results')  # 结果存储路径
MODEL_PATH = os.path.join(BASE_DIR, '..', 'yolo11n.pt')  # 模型路径

# 创建结果目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.debug(f"结果文件夹路径：{UPLOAD_FOLDER}")

# 视频处理相关配置
VIDEO_MAX_SIZE = 100 * 1024 * 1024  # 最大视频大小：100MB
VIDEO_PROCESSING = {}  # 存储视频处理状态 {video_id: {status, progress, result}}

# 加载YOLO模型
model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}（请将yolo11n.pt放在app.py同级目录）")

    # 新增：添加torch安全全局注册（解决权重加载兼容性问题）
    import torch
    from ultralytics.nn.tasks import DetectionModel

    torch.serialization.add_safe_global_class(DetectionModel)

    # 修改：显式设置weights_only=False，确保模型加载兼容
    model = YOLO(MODEL_PATH, weights_only=False)

    # 模型预热
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    test_results = model(test_img, verbose=False)
    if len(test_results) == 0 or not hasattr(test_results[0], 'boxes'):
        raise ValueError("模型推理异常")
    logging.debug("模型加载成功")
except Exception as e:
    logging.error(f"模型初始化失败：{str(e)}")
    model = None


def detect_image(image_path):
    """图片检测核心函数"""
    try:
        if not model:
            return {"error": "模型未加载"}

        if not os.path.exists(image_path):
            return {"error": f"原图不存在：{image_path}"}

        img = cv2.imread(image_path)
        if img is None:
            return {"error": "无法读取图片"}
        logging.debug(f"开始推理，原图尺寸：{img.shape}")

        # 模型推理
        t1 = time.time()
        results = model(img, conf=0.25, verbose=False)
        infer_time = round(time.time() - t1, 3)

        # 生成标注图片
        annotated_img = results[0].plot()
        timestamp = int(time.time())
        result_img_name = f"result_{timestamp}.jpg"
        result_img_path = os.path.join(UPLOAD_FOLDER, result_img_name)

        # 保存结果图片
        if not cv2.imwrite(result_img_path, annotated_img):
            return {"error": "无法保存结果图片"}
        logging.debug(f"结果图片保存成功：{result_img_path}")

        # 提取检测数据
        detect_data = []
        for idx, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            detect_data.append({
                "id": idx + 1,
                "class": results[0].names[cls_id],
                "confidence": round(float(box.conf[0]), 3),
                "coordinates": [round(float(c), 1) for c in box.xyxy[0]]
            })

        # 返回完整结果
        return {
            "result_img": f"/static/results/{result_img_name}",
            "file_name": result_img_name,
            "detect_data": detect_data,
            "infer_time": infer_time,
            "total_objects": len(detect_data)
        }

    except Exception as e:
        logging.error(f"检测失败：{str(e)}")
        return {"error": f"检测失败：{str(e)}"}


def process_video(video_path, video_id):
    """视频处理后台线程函数"""
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            VIDEO_PROCESSING[video_id] = {
                "status": "error",
                "message": "无法打开视频文件"
            }
            return

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.debug(f"开始处理视频，FPS: {fps}, 尺寸: {width}x{height}, 总帧数: {total_frames}")

        # 定义输出视频编码器和创建VideoWriter对象
        timestamp = int(time.time())
        result_video_name = f"result_video_{timestamp}.mp4"
        result_video_path = os.path.join(UPLOAD_FOLDER, result_video_name)

        # 使用MP4编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))

        # 存储每帧的检测数据
        frame_detections = []
        frame_count = 0
        start_time = time.time()

        # 逐帧处理
        while cap.isOpened():
            # 检查是否需要停止处理
            if VIDEO_PROCESSING.get(video_id, {}).get("status") == "cancelled":
                cap.release()
                out.release()
                if os.path.exists(result_video_path):
                    os.remove(result_video_path)
                VIDEO_PROCESSING[video_id] = {"status": "cancelled"}
                return

            ret, frame = cap.read()
            if not ret:
                break

            # 模型推理
            if model:
                results = model(frame, conf=0.25, verbose=False)[0]
                annotated_frame = results.plot()

                # 保存检测数据
                detections = []
                for idx, box in enumerate(results.boxes):
                    cls_id = int(box.cls[0])
                    detections.append({
                        "id": idx + 1,
                        "class": results.names[cls_id],
                        "confidence": round(float(box.conf[0]), 3),
                        "coordinates": [round(float(c), 1) for c in box.xyxy[0]]
                    })
                frame_detections.append({
                    "frame": frame_count,
                    "detections": detections
                })

                # 写入处理后的帧
                out.write(annotated_frame)

            # 更新进度
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            VIDEO_PROCESSING[video_id] = {
                "status": "processing",
                "progress": progress,
                "frame": frame_count,
                "total_frames": total_frames
            }

        # 释放资源
        cap.release()
        out.release()
        processing_time = round(time.time() - start_time, 2)

        # 计算总检测目标数
        total_objects = sum(len(frame["detections"]) for frame in frame_detections)

        # 更新处理状态
        VIDEO_PROCESSING[video_id] = {
            "status": "completed",
            "result_video": f"/static/results/{result_video_name}",
            "file_name": result_video_name,
            "processing_time": processing_time,
            "total_frames": total_frames,
            "total_objects": total_objects,
            "frame_detections": frame_detections
        }

        logging.debug(f"视频处理完成，耗时: {processing_time}秒，结果保存至: {result_video_path}")

    except Exception as e:
        error_msg = f"视频处理失败: {str(e)}"
        logging.error(error_msg)
        VIDEO_PROCESSING[video_id] = {
            "status": "error",
            "message": error_msg
        }


# 摄像头相关功能
camera_running = False
camera_thread = None


def generate_camera_frames():
    global camera_running
    cap = None
    try:
        cap = cv2.VideoCapture(0)  # 尝试打开摄像头
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logging.error("无法打开摄像头")
            return

        while camera_running:
            ret, frame = cap.read()
            if not ret:
                continue
            if model:
                results = model(frame, conf=0.25, verbose=False)[0]
                frame = results.plot()
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        logging.error(f"摄像头错误：{str(e)}")
    finally:
        if cap:
            cap.release()
        camera_running = False


@app.route('/camera_status')
def camera_status():
    return jsonify({"running": camera_running})


@app.route('/camera_feed')
def camera_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_running, camera_thread
    if not camera_running:
        camera_running = True
        camera_thread = threading.Thread(target=generate_camera_frames, daemon=True)
        camera_thread.start()
    return jsonify({"status": "started"})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running
    camera_running = False
    return jsonify({"status": "stopped"})


# 图片检测API
@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'file' not in request.files:
        return jsonify({"error": "未上传文件"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "请选择图片文件"}), 400

    # 检查文件格式
    allowed_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_ext:
        return jsonify({"error": f"仅支持{', '.join(allowed_ext)}格式"}), 400

    try:
        # 保存上传的图片
        timestamp = int(time.time())
        upload_filename = f"upload_{timestamp}{file_ext}"
        upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
        file.save(upload_path)
        logging.debug(f"上传图片保存：{upload_path}")

        # 调用检测函数
        result_data = detect_image(upload_path)
        if "error" in result_data:
            return jsonify({"error": result_data["error"]}), 500

        return jsonify(result_data)

    except Exception as e:
        return jsonify({"error": f"处理失败：{str(e)}"}), 500


# 视频检测相关API
@app.route('/api/detect_video', methods=['POST'])
def api_detect_video():
    if not model:
        return jsonify({"error": "模型未加载"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "未上传视频文件"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "请选择视频文件"}), 400

    # 检查文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > VIDEO_MAX_SIZE:
        return jsonify({"error": f"视频文件过大，最大支持{VIDEO_MAX_SIZE // (1024 * 1024)}MB"}), 400

    # 检查文件格式
    allowed_ext = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_ext:
        return jsonify({"error": f"仅支持{', '.join(allowed_ext)}格式"}), 400

    try:
        # 生成唯一视频ID
        video_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(os.urandom(4).hex())

        # 保存上传的视频
        upload_filename = f"upload_video_{video_id}{file_ext}"
        upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
        file.save(upload_path)
        logging.debug(f"上传视频保存：{upload_path}")

        # 初始化处理状态
        VIDEO_PROCESSING[video_id] = {
            "status": "processing",
            "progress": 0,
            "message": "开始处理视频..."
        }

        # 启动后台线程处理视频
        threading.Thread(
            target=process_video,
            args=(upload_path, video_id),
            daemon=True
        ).start()

        return jsonify({
            "video_id": video_id,
            "status": "processing_started"
        })

    except Exception as e:
        return jsonify({"error": f"视频处理启动失败：{str(e)}"}), 500


@app.route('/api/video_status/<video_id>')
def video_status(video_id):
    """查询视频处理状态"""
    status = VIDEO_PROCESSING.get(video_id, {"status": "not_found"})
    return jsonify(status)


@app.route('/api/cancel_video/<video_id>', methods=['POST'])
def cancel_video(video_id):
    """取消视频处理"""
    if video_id in VIDEO_PROCESSING:
        if VIDEO_PROCESSING[video_id]["status"] == "processing":
            VIDEO_PROCESSING[video_id]["status"] = "cancelled"
            return jsonify({"status": "cancelled"})
    return jsonify({"error": "视频处理不存在或已完成"}), 400


# 主页面
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# 下载结果文件
@app.route('/download/<filename>')
def download_file(filename):
    try:
        # 安全处理文件名（防止路径遍历攻击）
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

        # 验证文件存在性
        if not os.path.exists(file_path):
            logging.error(f"下载文件不存在：{file_path}")
            return "文件不存在", 404

        # 发送文件
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=safe_filename,
            mimetype='video/mp4' if safe_filename.endswith('.mp4') else 'image/jpeg'
        )

        # 下载后延迟删除（确保下载完成）
        @response.call_on_close
        def delete_after_download():
            # 延迟5秒删除，给大文件足够下载时间
            time.sleep(5)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logging.debug(f"下载后删除文件：{file_path}")
                except Exception as e:
                    logging.error(f"删除文件失败：{str(e)}")

        return response
    except Exception as e:
        logging.error(f"下载错误：{str(e)}")
        return "下载失败", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)