import cv2

from tqdm import tqdm

fname = 'train.txt'

classes = ['mosaic', 'line']

with open(fname, 'r', encoding='utf-8') as f:
    data = f.readlines()

for line in tqdm(data[:10]):
    try:
        img_path, bbox = line.split(' ')
        t, l, b, r, label = list(map(float, bbox.split(',')))
        image = cv2.imread(img_path)
        print(img_path)
        h, w = image.shape[:2]
        t = int(h*t)
        b = int(h*b) 
        l = int(w*l)
        r = int(w*r)
        cv2.rectangle(image, (l, t), (r, b), (0,0,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, classes[int(label)], (l,t),  font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow(fname, image)
        cv2.waitKey(0)
    except Exception as e:
        print(e)
        print(line)