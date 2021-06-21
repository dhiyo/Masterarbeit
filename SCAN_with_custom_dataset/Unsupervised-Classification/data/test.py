import os,re
from PIL import Image

image_name = os.listdir(r'C:\Users\shang\Documents\yueqi_ws\Masterarbeit\SCAN_with_custom_dataset\uiqa_Auschnitte\train')
target = []
data = []
for i in range(len(image_name)):
            num_index = re.search(r'\d', image_name[i]).start()
            label = image_name[i][:num_index]
            target.append(label)
            img_dir = 'C:/Users/shang/Documents/yueqi_ws/Masterarbeit/SCAN_with_custom_dataset/uiqa_Auschnitte/train/'+ image_name[i]
            img = Image.open(img_dir)
            data.append(img)
            print()

# import re
# s1 = "thishasadigit4here"
# m = re.search(r"\d", s1)
# if m:
#     print("Digit found at position", m.start())
# else:
#     print("No digit in that string")