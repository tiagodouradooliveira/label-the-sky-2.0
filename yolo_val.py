from ultralytics import YOLO
from config_loader import load_config
import os


paths = load_config("config/paths.json")
data_yaml_filename = "objdetdata.yaml"
split = "val"
model_dir = f"train"


classes = [0, 1, 2]


data_yaml_path = os.path.join(paths["root_path"], data_yaml_filename)

project_dir = os.path.join(paths["root_path"], "runs")

single_cls = False

batch = 256
conf = 0.001
iou = 0.5
max_det = 300

model = YOLO(paths['saved_models_path'] + "/" + model_dir + "/weights/best.pt")

metrics = model.val(split=split,
                    batch=batch, 
                    imgsz=128, 
                    save=True, 
                    device=-1, 
                    workers=8, 
                    project=project_dir, 
                    iou=iou,
                    single_cls=single_cls, 
                    classes=classes, 
                    max_det=max_det,
                    plots=True, 
                    augment=False,
                    half=False)