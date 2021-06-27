import sys
import pickle

data=[]
targets = []

file_path = r'C:\Users\shang\Documents\yueqi_ws\Masterarbeit\SCAN_with_custom_dataset\PNG-2-CIFAR10-master\uiqa.bin'
with open(file_path, 'rb') as f:
    if sys.version_info[0] == 2:
        entry = pickle.load(f)
    else:
        entry = pickle.load(f, encoding='latin1')
    data.append(entry['data'])
    if 'labels' in entry:
        targets.extend(entry['labels'])
    else:
        targets.extend(entry['fine_labels'])