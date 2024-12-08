!git clone https://github.com/ultralytics/yolov5  
%cd yolov5
!pip install -r requirements.txt  

import torch
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s')  

# imagem do dataset COCO
img_url = 'http://images.cocodataset.org/val2017/000000039769.jpg' 
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))

plt.imshow(img)
plt.axis('off')  # Ocultar os eixos
plt.show()

# Realizar a detecção de objetos na imagem
img = [img] 
results = model(img)  

# Exibir a imagem 
results.show()  

# resultados 
results.print() # resultados 