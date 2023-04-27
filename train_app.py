import sys
import os
sys.path.insert(0, os.path.dirname(os.getcwd()))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from ultralytics import YOLO

def train_model():
    # 加载模型
    # model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    print('model load。。。')
    model = YOLO("8npt/best.pt")  # 加载模型
    print('model load completed。。。')

    # 使用模型
    # model.train(data="img-layout.yaml", epochs=300, device=1)# , lr0=0.0001)  # 训练模型
    #
    # metrics = model.val()  # 在验证集上评估模型性能
    #
    # print('metric : {}'.format(metrics))

    # results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
    success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

if __name__ == '__main__':
    train_model()