from PIL import Image
import json
from tqdm import tqdm

json_file_dir =r'D:\dataset\Dataset format convert\Json files\via_region_data_train.json'
with open(json_file_dir) as f:
  my_data_dict = json.load(f)

num_images = len(my_data_dict['images'])
num_annotations = len(my_data_dict['annotations'])
for i in tqdm(range(num_annotations)):

    im_id = my_data_dict['annotations'][i]['image_id']
    file_name = my_data_dict['images'][im_id - 20180000001]['file_name']
    label_id = my_data_dict['annotations'][i]['category_id']
    label_name = my_data_dict['categories'][label_id - 1]['name']

    left = my_data_dict['annotations'][i]['bbox'][0]
    top = my_data_dict['annotations'][i]['bbox'][1]
    width = my_data_dict['annotations'][i]['bbox'][2]
    height = my_data_dict['annotations'][i]['bbox'][3]
    right = left + width
    bottom = top + height

    im = Image.open('uiqa_dataset/training/'+file_name)
    im = im.crop((int(left), int(top), int(right), int(bottom)))
    im.save('uiqa_schnitte/'+label_name+str(i)+'.png')




