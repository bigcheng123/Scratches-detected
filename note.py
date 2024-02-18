from utils.general import apply_classifier
from utils.datasets import LoadStreams, LoadImages

################# 此文件为测试用草稿纸 ############################

function datasets():
    class LoadStreams(source, img_size=imgsz, stride=stride)
        def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
    return self.sources, img, img0, None

dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    return self.sources, img, img0, None

function detect():
for path, img, im0s, vid_cap in dataset:

def apply_classifier(x, model, img, im0):
    return x
pred = apply_classifier(pred, modelc, img, im0s)

# Process detections
for i, det in enumerate(pred):  # detections per image
    if webcam:  # batch_size >= 1
        p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
    else:
        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)


res = cv2.resize(im0, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow(str(p), res)  ##### show images


pred = apply_classifier(pred, modelc, img, im0s)

for path, img, im0s, vid_cap in dataset:
