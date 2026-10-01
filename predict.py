from ultralytics import YOLO

# 官方训练模型
# yolo = YOLO(model='yolo26m.pt' , task='deteck')
# result = yolo(source = 'people.jpg',save=True)

# icon图标训练后模型
yolo = YOLO(model="icon.pt", task="deteck")
result = yolo(source="screen", save=True)
