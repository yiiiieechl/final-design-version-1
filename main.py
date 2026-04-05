import cv2
import tkinter as tk
from PIL import Image, ImageTk
import config
from detector import Detector
from spatial import spatial_filter
from temporal import TemporalJudge
from preprocess import preprocess_frame
import numpy as np
import time
import threading

# ====================== 全局配置 ======================
VIDEO_PATH = config.VIDEO_PATH
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

# 检测参数（每帧都检测，利用盈余性能）
DETECT_INTERVAL = 1

# ====================== 全局变量 ======================
cap = None
detector = None
temporal = None

# 视频原生素据
ORIGINAL_FPS = 24  # 视频原始帧率
FRAME_DELAY = 0    # 每帧应间隔时间（秒）

# 线程共享数据
data_lock = threading.Lock()
latest_frame = None
detect_result = []
face_keypoints = []
is_detecting = False
smoking_detected = False
detect_time = 0

# 统计
frame_count = 0
real_fps = 0        # 实际渲染帧率
total_smoke_count = 0
alert_count = 0

# 控制标志
system_running = True
is_recording = False
video_writer = None

# ====================== 初始化 ======================
def init_system():
    """系统初始化"""
    global cap, detector, temporal, ORIGINAL_FPS, FRAME_DELAY
    
    print("[INFO] 正在初始化系统...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] 无法打开视频文件")
        return False
    
    # 读取视频原生的帧率
    ORIGINAL_FPS = cap.get(cv2.CAP_PROP_FPS)
    if ORIGINAL_FPS <= 0:
        ORIGINAL_FPS = 24  # 默认值
    
    # 计算每帧应该显示多长时间（锁帧用）
    FRAME_DELAY = 1.0 / ORIGINAL_FPS
    print(f"[INFO] 视频原始帧率: {ORIGINAL_FPS}fps, 帧间隔: {FRAME_DELAY*1000:.1f}ms")
    
    # 初始化检测器
    detector = Detector(config.SMOKE_MODEL_PATH, config.POSE_MODEL_PATH)
    temporal = TemporalJudge(config.WINDOW_SIZE, config.TRIGGER_THRESHOLD)
    
    print("[INFO] 系统初始化完成，每帧都进行检测")
    return True

# ====================== 后台检测线程 ======================
def detection_thread():
    """后台检测线程（每帧都检测）"""
    global latest_frame, detect_result, face_keypoints
    global is_detecting, detect_time
    
    while system_running:
        if latest_frame is None or is_detecting:
            time.sleep(0.001)
            continue
        
        is_detecting = True
        frame = latest_frame.copy()
        
        # 开始计时
        start_time = time.time()
        
        # 预处理
        proc_frame = preprocess_frame(frame)
        
        # 模型推理（每帧都执行！）
        smoke_res, person_res, pose_res = detector.detect(
            proc_frame, config.SMOKE_CONF, config.PERSON_CONF
        )
        
        # 空间过滤
        valid_smokes = spatial_filter(proc_frame, smoke_res, person_res, pose_res)
        
        # 提取面部关键点（0-4号点：鼻子、左眼、右眼、左耳、右耳）
        faces = []
        if pose_res:
            for res in pose_res[:3]:  # 最多3个人
                try:
                    kpts = res.keypoints.xy[0].cpu().numpy()
                    face_points = kpts[0:5]  # 只取面部5个点
                    faces.append(face_points)
                except:
                    pass
        
        # 更新全局数据
        with data_lock:
            detect_result = valid_smokes
            face_keypoints = faces
            detect_time = (time.time() - start_time) * 1000
        
        is_detecting = False

# ====================== 主渲染循环 ======================
def video_loop():
    """视频渲染主循环（严格锁24fps）"""
    global frame_count, real_fps, latest_frame, smoking_detected
    global total_smoke_count, alert_count, system_running
    
    # 计算缩放比例
    video_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w_ratio = DISPLAY_WIDTH / video_w
    h_ratio = DISPLAY_HEIGHT / video_h
    
    last_time = time.time()
    last_frame_time = time.time()  # 用于锁帧
    
    while system_running:
        # ===== 锁帧控制：确保按原视频速度播放 =====
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        # 如果还没到下一帧的时间，就等待
        if elapsed < FRAME_DELAY:
            sleep_time = FRAME_DELAY - elapsed
            time.sleep(sleep_time)
        
        last_frame_time = time.time()  # 记录这一帧的开始时间
        
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last_frame_time = time.time()
            continue
        
        # 计算实际渲染帧率（用于显示）
        delta = time.time() - last_time
        real_fps = 1.0 / (delta + 0.0001)
        last_time = time.time()
        frame_count += 1
        
        # 共享给检测线程
        latest_frame = frame
        
        # 获取最新检测结果（无等待，直接取）
        with data_lock:
            smokes = detect_result.copy()
            faces = face_keypoints.copy() if face_keypoints else []
            inference_time = detect_time
        
        # 时序判断
        temporal.update(len(smokes) > 0)
        smoking_detected = temporal.get_ratio() > 0.5
        
        # ========== 绘制画面 ==========
        display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # 1. 画面部关键点（5个点）
        for face in faces:
            for i, (x, y) in enumerate(face):
                px = int(x * w_ratio)
                py = int(y * h_ratio)
                if px > 0 and py > 0:
                    # 鼻子大一点(4像素)，其他小一点(2像素)
                    radius = 4 if i == 0 else 2
                    # 鼻子绿色，眼睛青色，耳朵蓝色
                    if i == 0:
                        color = (0, 255, 0)      # 鼻子-绿
                    elif i <= 2:
                        color = (255, 0, 0)      # 眼睛-蓝  
                    else:
                        color = (255, 255, 0)    # 耳朵-青
                    cv2.circle(display, (px, py), radius, color, -1)
        
        # 2. 画香烟检测框（红色）
        for box in smokes:
            sx1, sy1, sx2, sy2, conf, _ = box
            x1 = int(sx1 * w_ratio)
            y1 = int(sy1 * h_ratio)
            x2 = int(sx2 * w_ratio)
            y2 = int(sy2 * h_ratio)
            
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display, f"{conf:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            total_smoke_count += 1
        
        # 3. 吸烟报警提示（红色闪烁）
        if smoking_detected:
            alert_count += 1
            if frame_count % 10 < 5:  # 闪烁
                cv2.putText(display, "SMOKING DETECTED!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.rectangle(display, (0, 0), (DISPLAY_WIDTH-1, DISPLAY_HEIGHT-1), (0, 0, 255), 3)
        
        # 4. 显示信息（左上角）
        cv2.putText(display, f"PlayFPS:{int(ORIGINAL_FPS)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"RealFPS:{int(real_fps)}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(display, f"Detect:{int(inference_time)}ms", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 5. 录制提示
        if is_recording:
            cv2.circle(display, (30, 30), 8, (0, 0, 255), -1)
            cv2.putText(display, "REC", (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if video_writer:
                video_writer.write(display)
        
        # 显示画面
        cv2.imshow("Smoking Detection System (24fps Locked)", display)
        
        # 按键处理（1ms等待不影响锁帧精度）
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            system_running = False
        elif key == ord(' '):
            pause_system()
        elif key == ord('r'):
            toggle_record()
        elif key == ord('s'):
            take_screenshot(display)

# ====================== 控制函数 ======================
def pause_system():
    """暂停/继续"""
    global system_running
    system_running = not system_running
    print(f"[INFO] 系统{'继续' if system_running else '暂停'}")

def toggle_record():
    """录制开关"""
    global is_recording, video_writer
    
    if not is_recording:
        filename = time.strftime("record_%Y%m%d_%H%M%S.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(filename, fourcc, int(ORIGINAL_FPS), (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        is_recording = True
        print(f"[INFO] 开始录制: {filename}")
    else:
        is_recording = False
        if video_writer:
            video_writer.release()
            video_writer = None
        print("[INFO] 停止录制")

def take_screenshot(frame):
    """截图"""
    filename = time.strftime("screenshot_%Y%m%d_%H%M%S.jpg")
    cv2.imwrite(filename, frame)
    print(f"[INFO] 截图已保存: {filename}")

# ====================== 控制面板 ======================
def create_control_panel():
    """Tkinter控制面板"""
    root = tk.Tk()
    root.title("控制面板")
    root.geometry("300x450+700+100")
    root.resizable(False, False)
    root.configure(bg='#f0f0f0')
    
    tk.Label(root, text="吸烟检测系统", font=("微软雅黑", 16, "bold"), 
            bg='#f0f0f0', fg='#333').pack(pady=10)
    
    # 状态框
    status_frame = tk.Frame(root, bg='#fff', bd=2, relief=tk.GROOVE)
    status_frame.pack(padx=10, pady=5, fill=tk.X)
    tk.Label(status_frame, text="系统状态", font=("微软雅黑", 10), bg='#fff').pack()
    status_label = tk.Label(status_frame, text="运行中", font=("Consolas", 12), 
                           bg='#fff', fg='green')
    status_label.pack()
    
    # FPS显示
    tk.Label(root, text="播放帧率(锁定)", font=("微软雅黑", 9), bg='#f0f0f0').pack()
    fps_label = tk.Label(root, text="24 FPS", font=("Consolas", 20, "bold"), 
                        bg='#f0f0f0', fg='#0066cc')
    fps_label.pack()
    
    # 实际帧率（盈余性能指标）
    real_fps_label = tk.Label(root, text="实际: 0 FPS", font=("Consolas", 10), 
                             bg='#f0f0f0', fg='#666')
    real_fps_label.pack()
    
    # 检测性能
    info_frame = tk.Frame(root, bg='#f0f0f0')
    info_frame.pack(pady=10)
    detect_label = tk.Label(info_frame, text="检测耗时: 0ms", font=("Consolas", 10), 
                           bg='#f0f0f0')
    detect_label.pack()
    count_label = tk.Label(info_frame, text="检测总数: 0", font=("Consolas", 10), 
                          bg='#f0f0f0')
    count_label.pack()
    alert_label = tk.Label(info_frame, text="报警次数: 0", font=("Consolas", 10), 
                          bg='#f0f0f0', fg='red')
    alert_label.pack()
    
    # 按钮
    btn_frame = tk.Frame(root, bg='#f0f0f0')
    btn_frame.pack(pady=15)
    tk.Button(btn_frame, text="暂停/继续", command=pause_system, 
             width=12, bg='#4CAF50', fg='white').pack(pady=2)
    tk.Button(btn_frame, text="录制", command=toggle_record, 
             width=12, bg='#f44336', fg='white').pack(pady=2)
    tk.Button(btn_frame, text="截图", command=lambda: take_screenshot(latest_frame), 
             width=12, bg='#2196F3', fg='white').pack(pady=2)
    
    tk.Label(root, text="快捷键: ESC退出 | 空格暂停 | R录制 | S截图", 
            font=("微软雅黑", 8), bg='#f0f0f0', fg='#666').pack(side=tk.BOTTOM, pady=10)
    
    # 更新UI
    def update_ui():
        if not system_running and frame_count > 100:  # 防止初始化时误判
            root.destroy()
            return
        
        with data_lock:
            current_infer = detect_time
            current_alert = alert_count
            current_total = total_smoke_count
            current_smoking = smoking_detected
            current_real_fps = real_fps
        
        detect_label.config(text=f"检测耗时: {int(current_infer)}ms")
        count_label.config(text=f"检测总数: {current_total}")
        alert_label.config(text=f"报警次数: {current_alert}")
        real_fps_label.config(text=f"实际: {int(current_real_fps)} FPS (盈余性能)")
        
        if current_smoking:
            status_label.config(text="检测到吸烟!", fg='red')
        else:
            status_label.config(text="运行中", fg='green')
        
        root.after(200, update_ui)
    
    update_ui()
    root.mainloop()

# ====================== 主函数 ======================
def main():
    """主函数"""
    if not init_system():
        return
    
    # 启动后台检测（每帧都检测）
    detect_thread = threading.Thread(target=detection_thread, daemon=True)
    detect_thread.start()
    print("[INFO] 后台检测已启动（每帧检测）")
    
    # 启动UI
    ui_thread = threading.Thread(target=create_control_panel, daemon=True)
    ui_thread.start()
    
    # 主线程跑视频（严格锁24fps）
    print("[INFO] 开始播放（锁定24fps，盈余性能用于检测）...")
    try:
        video_loop()
    except Exception as e:
        print(f"[ERROR] 异常: {e}")
    
    # 清理
    global system_running
    system_running = False
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 系统已关闭")

if __name__ == "__main__":
    main()