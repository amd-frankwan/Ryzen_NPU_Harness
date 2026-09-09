import random
import requests
from pycocotools.coco import COCO

# Initialize COCO API
annFile = "annotations/instances_val2017.json"
coco = COCO(annFile)

# Get 100 random image IDs
imgIds = coco.getImgIds()
random_img_ids = random.sample(imgIds, 100)
images_info = coco.loadImgs(random_img_ids)

# Download images
for img_info in images_info:
  url = img_info["coco_url"]
  response = requests.get(url)
  with open(img_info["file_name"], "wb") as f:
    f.write(response.content)