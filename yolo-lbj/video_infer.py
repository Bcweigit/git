import cv2
from ultralytics import YOLO
import os

def video_detection(video_path, model_path, output_path, conf_thres=0.4, show_result=True):
    """
    对输入视频进行零部件检测
    :param video_path: 输入视频路径
    :param model_path: 模型权重路径
    :param output_path: 结果视频保存路径
    :param conf_thres: 置信度阈值
    :param show_result: 是否实时弹窗显示结果
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件，请检查路径")
        return
    
    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"开始检测，共 {total_frames} 帧，请稍候...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO推理
        results = model(frame, conf=conf_thres, verbose=False)
        
        # 绘制检测框、类别、置信度
        annotated_frame = results[0].plot()
        
        # 写入结果视频
        out.write(annotated_frame)
        frame_count += 1
        
        # 实时显示
        if show_result:
            cv2.imshow('Parts Detection', annotated_frame)
            # 按 q 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 打印进度
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n检测完成，共处理 {frame_count} 帧")
    print(f"结果视频已保存至：{output_path}")

# ========== 本地测试入口 ==========
if __name__ == '__main__':
    # 把下面的视频路径改成你自己的视频文件路径
    video_detection(
        video_path='求取零部件的角度.mp4',      # 待检测的视频文件路径
        model_path='model/best.pt', # 模型权重路径
        output_path='result_output.mp4', # 结果视频保存路径
        conf_thres=0.40,             # 置信度阈值
        show_result=True            # 是否实时弹窗显示
    )
