from ultralytics import YOLO

#load model
#初始训练模型
#model = YOLO("yolo26m.pt" )  

#中断后再次使用的训练模型
model = YOLO(r'.\runs\detect\train-7\weights\last.pt')  #last.pt为最后一个result的生成结果

#train model
#epoch:训练轮数  batch:每次训练的图片数量  workers:加载数据的线程数  data:数据集配置文件
#model.train(data = 'icon.yaml',workers=0,epochs=300,batch=16) 

#中断后再次训练使用的模型
model.train(resume=True)