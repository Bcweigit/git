from ultralytics import YOLO

def main():
    # 加载预训练模型
    # 速度优先选 yolov8n.pt；精度优先可换成 yolov8s.pt / yolov8m.pt
    model = YOLO('yolov8n.pt')

    # 启动训练
    results = model.train(
        data='jc.yaml',          # 你的数据集配置文件
        epochs=100,              # 训练总轮次，小数据集可先跑 50 轮观察收敛情况
        imgsz=640,               # 输入图片尺寸
        batch=16,                # 批次大小，显存不足就调小（8、4）
        device=0,                # 使用第 0 块 GPU；无 GPU 填 'cpu'
        workers=4,               # 数据加载线程数
        project='runs/train',    # 训练结果保存目录
        name='parts_det',        # 本次实验文件夹名
        patience=20,             # 早停策略：连续 20 轮精度不提升就自动停止
        save=True,
        plots=True,               # 自动生成精度、损失曲线等可视化图
        channels_last=False,
        amp=False
    )

if __name__ == '__main__':
    main()
