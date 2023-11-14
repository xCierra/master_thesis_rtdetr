import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    model.load('/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/weights/rtdetr-l.pt') # loading pretrain weights
    model.train(data='/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/dataset/classroom_data.yaml',
                cache=False,
                imgsz=640,
                epochs=500,
                batch=64,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/outputs/train',
                name='rtdetr-l-64bs-500ep',
                )