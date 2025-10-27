from ultralytics import YOLO
from config_loader import load_config
import os


paths = load_config("config/paths.json")
data_yaml_filename = "objdetdata.yaml"


classes = [0, 1, 2]


data_yaml_path = os.path.join(paths["root_path"], data_yaml_filename)

project_dir = os.path.join(paths["root_path"], "runs")

single_cls = False

seed = 42
epochs = 500
patience = 20
batch = 256
optimizer = "Adam"
lr0 = 0.00976
lrf = 0.0094
momentum = 0.94251
warmup_epochs = 2.29867
warmup_momentum = 0.60694
weight_decay = 0.0
box = 4.72755
cls = 0.42222
dfl = 1.07724
degrees = 27.6169
flipud = 0.25862
fliplr = 0.35492
nbs = 256
dropout = 0.0
scale = 0.0
mosaic = 0.0
hsv_h = 0.0
hsv_s = 0.0
hsv_v = 0.0
translate = 0.0
shear = 0.0
perspective = 0.0
bgr = 0.0
mosaic = 0.0
mixup = 0.0
cutmix = 0.0
copy_paste = 0.0
erasing = 0.0

model = YOLO("yolov3.yaml")

model.train(data=data_yaml_path, 
            epochs=epochs, 
            patience=patience, 
            batch=batch, 
            imgsz=128, 
            save=True, 
            device=-1, 
            workers=8, 
            project=project_dir, 
            exist_ok=False, 
            pretrained=False, 
            optimizer=optimizer, 
            seed=seed, 
            deterministic=True,
            single_cls=single_cls, 
            classes=classes, 
            resume=False, 
            amp=False, 
            lr0=lr0, 
            lrf=lrf, 
            momentum=momentum,
            weight_decay=weight_decay, 
            box=box, 
            cls=cls, 
            dfl=dfl, 
            nbs=nbs, 
            dropout=dropout, 
            val=True, 
            plots=True, 
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            hsv_h=hsv_h, 
            hsv_s=hsv_s, 
            hsv_v=hsv_v, 
            bgr=bgr,
            degrees=degrees, 
            translate=translate, 
            scale=scale, 
            shear=shear, 
            perspective=perspective, 
            flipud=flipud, 
            fliplr=fliplr, 
            mosaic=mosaic, 
            mixup=mixup, 
            cutmix=cutmix, 
            copy_paste=copy_paste, 
            erasing=erasing)
