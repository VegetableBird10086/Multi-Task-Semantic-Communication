import torch
import torchvision
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)
model = model.to('cuda:5')
mode = 'test'
with open('data/mmimdb/split.json', 'r') as file:
    # Load the JSON data
    data = json.load(file)
images_ids  = data[mode]
fast = {}
with torch.no_grad():

    model.eval()
    for img_id in tqdm(images_ids):
        img_path = 'data/mmimdb/dataset/' + img_id + '.jpeg'
        image = Image.open(img_path) 
        image = image.convert('RGB')
        draw = ImageDraw.Draw(image)

        transformed_img = torchvision.transforms.transforms.ToTensor()(image)
        transformed_img = transformed_img.to('cuda:5')
        result = model([transformed_img])
        boxes = result[0]['boxes'][:15]
        boxes = boxes.cpu().detach().numpy().tolist()
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                boxes[i][j] = int(boxes[i][j])
        fast[img_id] = boxes


with open('data/mmimdb/%s_fast.json' % mode, 'w') as f:
    json.dump(fast, f)
        

    

    
            