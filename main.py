from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from ui_files.main_win import Ui_mainWindow
from ui_files.dialog.rtsp_win import Window

from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from pathlib import Path
import sys
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2
import modbus_rtu
import _thread

from shutil import copy

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam, LoadStreams
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path,clean_str
# from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.plots import Annotator, colors, save_one_box,plot_one_box

from utils.torch_utils import select_device,time_sync,load_classifier
from utils.capnums import Camera

## set global variable 设置全局变量
modbus_flag = False
results =[]

class DetThread(QThread): ###继承 QThread
    send_img_ch0 = pyqtSignal(np.ndarray)  ### CH0 output image
    send_img_ch1 = pyqtSignal(np.ndarray)  ### CH1 output image
    send_img_ch2 = pyqtSignal(np.ndarray)  ### CH2 output image
    send_img_ch3 = pyqtSignal(np.ndarray)  ### CH3 output image
    send_img_ch4 = pyqtSignal(np.ndarray)  ### CH4 output image
    send_img_ch5 = pyqtSignal(np.ndarray)  ### CH5 output image
    send_statistic = pyqtSignal(dict)  ###
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.vid_cap = None #240229
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.device = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = False                # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = None  ####'./auto_save/jpg'

    @torch.no_grad()
    def run(self,
            imgsz=640, #1440 # inference size (pixels)//推理大小
            max_det=50,  # maximum detections per image//每个图像的最大检测次数
            # self.source = '0'
            # self.device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)//边界框厚度
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=True,  # use FP16 half-precision inference
            ):

        save_img = not nosave and not self.source.endswith('.txt')  # save inference images
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        # try:
        set_logging()
        device = select_device(self.device)  ### from utils.torch_utils import select_device
        half &= device.type != '0'  #'cpu'# half precision only supported on CUDA

        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model  from models.experimental import attempt_load
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False  ### TODO:bug-3  can not set True
        if classify:
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

        # Dataloader
        if webcam: ###self.source.isnumeric() or self.source.endswith('.txt') or
            print('if webcam is running')
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)  #### loadstreams  return self.sources, img, img0, None
            print('dataset type', type(dataset), dataset)
            bs = len(dataset)  # batch_size
            print('len(dataset)=', bs)
            # #### streams = LoadStreams

        else:  ### load the images
            print('if webcam false')
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs


        # Run inference 推理
        if device.type != '0':#'cpu'
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        start_time = time.time()
        t0 = time.time()
        count = 0

        # load  the camera's index 载入 摄像头号码
        streams_list = []
        with open('streams.txt', 'r') as file:
            for line in file:
                streams_list.append(line.strip())
        print('streams:', streams_list)

        # dataset = iter(dataset)  ##迭代器 iter 创建了一个迭代器对象，每次调用这个迭代器对象的__next__()方法时，都会调用 object
        while True: ##### 采用循环来 检查是否 停止推理
            print('marker while loop')
            print(' while loop self.is_continue', self.is_continue)
            print(' while loop self.jump_out', self.jump_out)
            # print(' while loop camera.cap', type(self.vid_cap))

            if self.jump_out:
                # LoadStreams.release_camera()
                self.vid_cap.release()  #### TODO: bug-2  无法释放所有摄像头 ，只能释放追后一个摄像头资源
                print('vid_cap.release -1', type(self.vid_cap))
                LoadStreams.streams_update_flag = False
                self.send_percent.emit(0)
                self.send_msg.emit('Stop')
                if hasattr(self, 'out'):
                    self.out.release()
                print('jump_out push-11', self.jump_out)
                break

            # change model & device  20230810
            if self.current_weight != self.weights:
                print('current is running')
                # Load model
                model = attempt_load(self.weights, map_location = device)  # load FP32 model
                stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(imgsz, s=stride)  # check image size
                names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                if half:
                    model.half()  # to FP16将模型的权重参数和激活值的数据类型转换为半精度浮点数格式。
                    ### 这种转换可以减少模型的内存占用和计算开销，从而提高模型在 GPU 上的运行效率
                # Run inference
                if device.type != '0':#'cpu'
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                self.current_weight = self.weights


            # load  streams
            pred_flag = False

            if self.is_continue:
                print('is continue is running')
                print('DetThread.run.is_continue : true')

                #  loadstreams // dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
                for path, img, im0s, self.vid_cap in dataset:  # 由于dataset在RUN中运行 会不断更新，所以此FOR循环 不会穷尽

                    # print(type(path), type(img), type(im0s), type(self.vid_cap))
                    # #for testing : show row image
                    # cv2.imshow('ch0', im0s[0])
                    # cv2.imshow('ch1', im0s[1])
                    # ### img recode
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    statistic_dic = {name: 0 for name in names}  # made the diction
                    # print('statisstic_dic-1',statistic_dic)
                    count += 1  # ### FSP counter
                    if  count % 30 == 0 and count >= 30:
                        loopcycle = int(30/(time.time()-start_time))  #### 大循环周期
                        self.send_fps.emit('fps：'+str(loopcycle))
                        start_time = time.time() # updata start-time
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    t2 = 0
                    # if not pred_frag  output raw frame
                    if not pred_flag:  # # todo bug-4 建立2种图像输出方式 ， 原始输出  VS  预测结果后输出，控制变量 = pred_flag
                        for i, index in enumerate(streams_list):
                            t1 = time.time()
                            label_chanel = str(streams_list[i])
                            print(i, index, label_chanel)
                            im0 = im0s
                            # cv2.imshow('ch0', im0s[0])
                            # cv2.imshow('ch1', im0s[1])
                            # send img :
                            # print('detect_cycle', detect_cycle)
                            # time.sleep(0.2)
                            # # detect_cycle = 30
                            # timer = int(1 / (t1 - t2))
                            # print(timer)  # todo  CV2格式图像 需要转换为  QLabel 格式才能 emit
                            # cv2.putText(im0, str(f'FSP = {detect_cycle}  CAM = {label_chanel}'), (20, 30),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                            # res = cv2.resize(im0, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            # ## chanel-0  ##### show images
                            # if label_chanel == '0':
                            #     self.send_img_ch0.emit(im0)  ### 发送图像
                            #     print('seng img : ch0')
                            # ## chanel-1
                            # if label_chanel == '1':
                            #     self.send_img_ch1.emit(im0)  ### 发送图像
                            #     # print('seng img : ch1')
                            # # chanel-2
                            # if label_chanel == '2':
                            #     self.send_img_ch2.emit(im0)  ### 发送图像fi
                            #     # print('seng img : ch2')
                            # ## chanel-3
                            # if label_chanel == '3':
                            #     self.send_img_ch3.emit(im0)  #### 发送图像
                            #     # print('seng img : ch3')
                            # ## chanel-4
                            # if label_chanel == '4':
                            #     self.send_img_ch4.emit(im0)  #### 发送图像
                            #     # print('seng img : ch4')
                            # ## chanel-5
                            # if label_chanel == '5':
                            #     self.send_img_ch5.emit(im0)  #### 发送图像
                            #     # print('seng img : ch5')
                            # ### ## send the detected result
                            # self.send_statistic.emit(statistic_dic)  # 发送 检测结果 statistic_dic
                            # # print('emit statistic_dic', statistic_dic)
                            t2 = time.time()
                    # Inference prediction
                    if pred_flag:  # TODO ： 原来的代码  预测后输出
                        # pred = model(img, augment=augment)[0] #### 预测  使用loadWebcam是 加载的model
                        pred = model(img,
                                     augment=augment,
                                     visualize=increment_path(save_dir / Path(path).stem,
                                                              mkdir=True) if visualize else False)[0]

                        # Apply NMS
                        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)

                        # Apply Classifier
                        if classify: #classify = False
                            pred = apply_classifier(pred, modelc, img, im0s)
                            # print(f'type pred = ', type(pred), len(pred))

                        # emit frame  & Process detections
                        for i, det in enumerate(pred):  # detections per image
                            # ## label_index 方法1  ↓ ###label_chanel 依据 list det的 元素
                            # label_chanel = str(i)
                            ### label_index 方法2  ↓  依据 streams.txt camera号码
                            if len(pred) <= len(streams_list):
                                label_chanel = str(streams_list[i])
                            else:
                                print(f'streams : {len(pred)} camera quantity : {len(streams_list)}')
                                break
                            # print(type(label_chanel),'img chanel=', label_chanel)

                            if webcam:  # batch_size >= 1     get the frame
                                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                            else: ### image
                                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                            p = Path(p)  # to Path
                            # save_path = str(save_dir / p.name)  # img.jpg
                            # txt_path = str(save_dir / 'labels' / p.stem) + (dtxt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  #
                            txt_path = 'auto_save'
                            s += '%gx%g ' % img.shape[2:]  # print string
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            imc = im0.copy() #if save_crop else im0  # for save NG frame
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                                # save_txt / plot one box /save image
                                for *xyxy, conf, cls in reversed(det):
                                    if save_txt:  # save_txt=False,  # save results to *.txt
                                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                            -1).tolist()  # normalized xywh
                                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                        with open(txt_path + '.txt', 'a') as f:
                                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                    # plot_one_box here
                                    if save_img or save_crop or view_img:  # Add bbox to image
                                        # save_img = not nosave and not self.source.endswith('.txt')  # save inference images
                                        # save_crop=False,  # save cropped prediction boxes
                                        # view_img =check_inshow() # Check if environment supports image displays
                                        # print(f'Line 317 save_img {save_img},save_crop {save_crop},view_img {view_img}')
                                        c = int(cls)  # integer class
                                        statistic_dic[names[c]] += 1
                                        # print('statisstic_dic-2',statistic_dic)
                                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                        plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                                     line_thickness=line_thickness)
                                        #### save NG image at here

                                        # auto_save  Write results  save NG image in floder jpg
                                        global results
                                        if self.save_fold:  #### when checkbox: autosave is  setcheck
                                            os.makedirs(self.save_fold, exist_ok=True)
                                            if len(det) :
                                                if names[c] == 'impress':  # 限定保存类型
                                                    save_path = os.path.join(self.save_fold,
                                                                             f'{names[c]}_' + time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                                           time.localtime()) + f'_Cam{label_chanel}' + '.jpg')
                                                    # cv2.imwrite(save_path, im0)  # im0 = im0s.copy()  with box
                                                    cv2.imwrite(save_path, imc)  # imc = no box
                                                    print(str(f'save as .jpg im{i} , CAM = {label_chanel},save_path={save_path}'))  # & str(save_path))
                                                    # print(f'class name {names[c]},type{type(names[c])}')
                                        if save_crop:
                                            print('save_one_box')
                                # print('detection is running')


                            # print(f'{s}Done. ({t2 - t1:.3f}s detect_cycle={detect_cycle})')
                            # precition end #######################################################################


                            #   emit frame  Stream results

                            if self.is_continue: ###### send image in loop @  for i, det in enumerate(pred):
                                t2 = time_sync()
                                detect_cycle = int(1 / (t2 - t1))
                            # send img :
                                cv2.putText(im0, str(f'FSP = {detect_cycle}  CAM = {label_chanel}'), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                                res = cv2.resize(im0, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                                ## chanel-0  ##### show images
                                if label_chanel == '0':
                                    self.send_img_ch0.emit(im0)  ### 发送图像
                                    # print('seng img : ch0')
                                ## chanel-1
                                if label_chanel == '1':
                                    self.send_img_ch1.emit(im0)  ### 发送图像
                                    # print('seng img : ch1')
                                # chanel-2
                                if label_chanel == '2':
                                    self.send_img_ch2.emit(im0)  ### 发送图像fi
                                    # print('seng img : ch2')
                                ## chanel-3
                                if label_chanel == '3':
                                    self.send_img_ch3.emit(im0)  #### 发送图像
                                    # print('seng img : ch3')
                                ## chanel-4
                                if label_chanel == '4':
                                     self.send_img_ch4.emit(im0)  #### 发送图像
                                     # print('seng img : ch4')
                                ## chanel-5
                                if label_chanel == '5':
                                     self.send_img_ch5.emit(im0)  #### 发送图像
                                     # print('seng img : ch5')
                                ### ## send the detected result
                                self.send_statistic.emit(statistic_dic)  #发送 检测结果 statistic_dic
                                # print('emit statistic_dic', statistic_dic)
                    '''
                    if self.save_fold:  #### when checkbox: autosave is  setcheck
                        # save as mp4
                        if self.vid_cap is None:  ####save as .mp4
                            # else: ### self.vid_cap is cv2capture save as .mp4
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                                       time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                            print(str(f'save as .mp4  CAM = {label_chanel}'))  # & str(save_path))
                    '''
                    if self.rate_check:
                        time.sleep(1/self.rate)
                    # im0 = annotator.result()

                    if self.rate_check:
                        time.sleep(1/self.rate)
                    # im0 = annotator.result()
                    # Write results

                    if self.jump_out:
                        print('jump_out push-2', self.jump_out)
                        self.is_continue = False
                        # cap1 = cv2.VideoCapture(0)
                        # cap2 = cv2.VideoCapture(1)
                        # cap3 = cv2.VideoCapture(2)
                        # cap4 = cv2.VideoCapture(3)
                        # cap5 = cv2.VideoCapture(4)
                        # cap6 = cv2.VideoCapture(5)
                        # cap1.release()
                        # print('capr', cap1.release())
                        # cap2.release()
                        # print('cap2r', cap2.release())
                        # cap3.release()
                        # print('cap3r', cap3.release())
                        # cap4.release()
                        # print('cap4r', cap4.release())
                        # cap5.release()
                        # print('cap5r', cap5.release())
                        # cap6.release()
                        # print('cap6r', cap6.release())
                        # # print('cap.is_open', self.cap.isOpened())
                        # self.send_percent.emit(0)
                        # self.send_msg.emit('Stop')
                        # if hasattr(self, 'out'):
                        #     self.out.release()
                        #     print('self.out.release')
                        # break
                        '''
                        c = 2
                        while c > -1:
                            print('2-vid_cap', self.vid_cap)
                            if self.vid_cap is not None:
                                self.vid_cap.release() # 释放视频捕获对象
                                print('2-in loop', c, self.vid_cap.release(), self.vid_cap)
                            else:
                                print('2-in loop', c, 'vid_cap is None')
                            c -= 1
                            if c >= 0:
                                self.vid_cap = cv2.VideoCapture(c)
                                print('2-fix cap', self.vid_cap)
                            else:
                                print('2-fix cap None')
                        if self.vid_cap is None:
                            self.send_percent.emit(c)
                            self.send_msg.emit('Stop')
                            if hasattr(self, 'out'):
                                self.out.release()
                                print('self.out released')
                            break
                        '''
                        #     if c == -1:
                        #         self.vid_cap = None
                        #         # self.jump_out = False
                        #         # self.is_continue = False
                        #         self.send_percent.emit(c)
                        #         self.send_msg.emit('Stop')
                        #         if hasattr(self, 'out'):
                        #             self.out.release()
                        #             print('self.out.release')
                        #         break
                        #         print('2-reset cap', self.vid_cap)
                        # break

                        self.vid_cap.release()  #### bug-2  无法释放摄像头  未解决
                        print('self.vid_cap.release-22', type(self.vid_cap))
                        self.send_percent.emit(0)
                        self.send_msg.emit('Stop')
                        if hasattr(self, 'out'):
                            self.out.release()
                            print('self.out.release')
                        break


                if percent == self.percent_length:
                    print(count)
                    self.send_percent.emit(0)
                    self.send_msg.emit('finished')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break
            else:
                print('is_continue break', self.is_continue)

        #### 生成结果文件夹
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     print(f"Results saved to {save_dir}{s}")

        if update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)

        # except Exception as e:
        #     self.send_msg.emit('%s' % e)



class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # self.LoadStreams_thread = None
        self.setupUi(self)
        self.m_flag = False

        # style 1: window can be stretched
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # Transparency of window

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # search models automatically
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)

        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()  ### get model from combobox
        self.device_type = self.comboBox_device.currentText()  ###  get device type from combobox
        self.source_type = self.comboBox_source.currentText()  ###  get device type from combobox
        self.port_type = self.comboBox_port.currentText() ###  get port type from combobox
        self.det_thread.weights = "./pt/%s" % self.model_type  # difined
        self.det_thread.device = self.device_type # difined  device
        self.det_thread.source = self.source_type # get origin source index
        self.det_thread.percent_length = self.progressBar.maximum()
        #### the connect funtion transform to  def run_or_continue(self):
        #### tab1-mutil
        self.det_thread.send_img_ch0.connect(lambda x: self.show_image(x, self.video_label_ch0))
        self.det_thread.send_img_ch1.connect(lambda x: self.show_image(x, self.video_label_ch1))
        self.det_thread.send_img_ch2.connect(lambda x: self.show_image(x, self.video_label_ch2))
        self.det_thread.send_img_ch3.connect(lambda x: self.show_image(x, self.video_label_ch3))
        self.det_thread.send_img_ch4.connect(lambda x: self.show_image(x, self.video_label_ch11))
        self.det_thread.send_img_ch5.connect(lambda x: self.show_image(x, self.video_label_ch12))
        #### tab-2
        self.det_thread.send_img_ch0.connect(lambda x: self.show_image(x, self.video_label_ch4))
        #### tab-3
        self.det_thread.send_img_ch1.connect(lambda x: self.show_image(x, self.video_label_ch5))
        #### tab-4
        self.det_thread.send_img_ch2.connect(lambda x: self.show_image(x, self.video_label_ch6))
        #### tab-5
        self.det_thread.send_img_ch3.connect(lambda x: self.show_image(x, self.video_label_ch7))
        #### tab-6
        self.det_thread.send_img_ch4.connect(lambda x: self.show_image(x, self.video_label_ch8))
        #### tab-7
        self.det_thread.send_img_ch5.connect(lambda x: self.show_image(x, self.video_label_ch9))

        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.runButton_modbus.clicked.connect(self.modbus_on_off)
        self.testButton.clicked.connect(self.testfuntion)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.comboBox_device.currentTextChanged.connect(self.change_device)
        self.comboBox_source.currentTextChanged.connect(self.change_source)
        self.comboBox_port.currentTextChanged.connect(self.change_port)

        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()  #### loading config



    def run_or_continue(self):
        # self.det_thread.source = 'streams.txt'
        self.det_thread.jump_out = False
        print('runbutton is check', self.runButton.isChecked())
        if self.runButton.isChecked():
            self.runButton.setText('PAUSE')
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            device = os.path.basename(self.det_thread.device)  ### only for display
            source = os.path.basename(self.det_thread.source)  ### 引用 det_thread类的 self.source
            source = str(source) if source.isnumeric() else source  ### source为 int时 转换为 str
            self.statistic_msg('Detecting >> model：{}，device: {}, source：{}'.
                               format(os.path.basename(self.det_thread.weights),device,
                                      source))
            print('self.det_thread.is_continue', self.det_thread.is_continue)

        else:
            self.det_thread.is_continue = False
            self.runButton.setText('RUN')
            self.statistic_msg('Pause')
            print('self.det_thread.is_continue', self.det_thread.is_continue)

    def testfuntion(self):
        print('------------------------------testfuntion button push')
        # self.LoadStreams_thread = LoadStreams()  ####

        # if LoadStreams.streams_update_flag:
        #     self.LoadStreams_thread.streams_update_flag = False
        #
        # # print('streams_update_flag', self.LoadStreams_thread.streams_update_flag)
        # else:
        #     # LoadStreams.streams_update_flag = False
        #     self.LoadStreams_thread.streams_update_flag = True
        #     # print('streams_update_flag', self.LoadStreams_thread.streams_update_flag)

    def thread_mudbus_run(self):
        global modbus_flag
        modbus_flag = True
        # hexcode   comunicate with PC and modbus device
        # IN0_READ = '01 02 00 00 00 01 B9 CA'
        # IN1_READ = '01 02 00 01 00 01 E8 0A'
        # IN2_READ = '01 02 00 02 00 01 18 0A'
        # IN3_READ = '01 02 00 03 00 01 49 CA'
        DO0_ON = '01 05 33 0A FF 00 A3 7C'#地址330A-Y12接通亮黄灯 原代码：'01 05 00 00 FF 00 8C 3A'
        DO0_OFF = '01 05 33 0A 00 00 E2 8C'#地址330A-Y12断开灭黄灯 原代码：'01 05 00 00 00 00 CD CA'
        # DO1_ON = '01 05 00 01 FF 00 DD FA'
        # DO1_OFF = '01 05 00 01 00 00 9C 0A'
        DO2_ON = '01 05 33 0C FF 00 43 7D'#地址330C-Y14接通亮绿灯 原代码：'01 05 00 02 FF 00 2D FA'
        DO2_OFF = '01 05 33 0C 00 00 02 8D'#地址330C-Y14断开灭绿灯 原代码：'01 05 00 02 00 00 6C 0A'
        DO3_ON = '01 05 33 0B FF 00 F2 BC'#地址330B-Y13接通亮红灯 原代码：'01 05 00 03 FF 00 7C 3A'
        DO3_OFF = '01 05 33 0B 00 00 B3 4C'#地址330B-Y13断开灭红灯 原代码：'01 05 00 03 00 00 3D CA'

        DO_ALL_ON = '01 0F 00 00 00 04 01 FF 7E D6'
        DO_ALL_OFF = '01 0F 33 0A 00 03 01 00 12 95'#'01 0F 00 00 00 04 01 00 3E 96' ##OUT1-4  OFF  全部继电器关闭  初始化

        # self.ret = None
        self.port_type = self.comboBox_port.currentText()
        print(type(self.port_type), self.port_type)
        # try:
        #     self.ser, self.ret, _ = modbus_rtu.openport(port='COM5', baudrate=9600, timeout=5)  # 打开端口
        #     print('self.ret',self.ret)
        # except Exception as e:
        #     self.ret = False
        #     print('open port error', e)
        #     self.statistic_msg(str(e))
        #     self.runButton_modbus.setChecked(False)

        if self.ret: ### openport sucessfully
            feedback_data = modbus_rtu.writedata(self.ser, DO_ALL_OFF)  ###OUT1-4  OFF  全部继电器关闭  初始化
            self.runButton_modbus.setChecked(True)
            print('thread_mudbus_run modbus_flag = True')
            feedback_list = []

            while self.runButton_modbus.isChecked() and modbus_flag:
                start = time.time()
                # 240228屏蔽537-595:速度提升0.26s
                # feedback_data_in0 = modbus_rtu.writedata(self.ser, IN0_READ)  #### 检查IN1 触发 返回01 02 01 00 a188
                # if feedback_data_in0:#### 有返回数据
                #     text_in0 = feedback_data_in0[0:8]  ## 读取8位字符
                #     if text_in0 == '01020101':
                #         self.checkBox_10.setChecked(True)
                #     else:
                #         self.checkBox_10.setChecked(False)
                #     print('text_IN0', text_in0)
                #     feedback_list.append(text_in0)
                #     # feedback_data = modbus_rtu.writedata(self.ser, DO0_ON) ###1号继电器打开  运行准备 DO1 =1
                #     # feedback_data = modbus_rtu.writedata(self.ser, DO2_ON)  ###PLC控制，亮绿灯-240228
                # else: #### 无返回数据
                #     no_feedback = modbus_rtu.writedata(self.ser, DO2_ON)  ###3号继电器打开   控制器无返回数据 D03 =1
                #     print('no_feedback data')
                #
                # feedback_data_in1 = modbus_rtu.writedata(self.ser, IN1_READ)  #### 检查IN2 触发 返回01 02 01 00 a188
                # if feedback_data_in1:  #### 有返回数据
                #     text_in1 = feedback_data_in1[0:8]  ## 读取8位字符
                #     if text_in1 == '01020101':
                #         self.checkBox_11.setChecked(True)
                #     else:
                #         self.checkBox_11.setChecked(False)
                #     print('text_IN1', text_in1)
                #     feedback_list.append(text_in1)
                # else:  #### 无返回数据
                #     no_feedback = modbus_rtu.writedata(self.ser,DO2_ON)  ###3号继电器打开   控制器无返回数据 D03 =1
                #     print('no_feedback data')
                #
                # feedback_data_in2 = modbus_rtu.writedata(self.ser,IN2_READ)  #### 检查IN2 触发 返回01 02 01 00 a188
                # if feedback_data_in2:  #### 有返回数据
                #     text_in2 = feedback_data_in2[0:8]  ## 读取8位字符
                #     if text_in2 == '01020101':
                #         self.checkBox_12.setChecked(True)
                #     else:
                #         self.checkBox_12.setChecked(False)
                #     print('text_IN2', text_in2)
                #     feedback_list.append(text_in2)
                # else:  #### 无返回数据
                #     no_feedback = modbus_rtu.writedata(self.ser,DO2_ON)  ###3号继电器打开   控制器无返回数据 D03 =1
                #     print('no_feedback data')
                #
                # feedback_data_in3 = modbus_rtu.writedata(self.ser,IN3_READ)  ####
                # if feedback_data_in3:  #### 有返回数据
                #     text_in3 = feedback_data_in3[0:8]  ## 读取8位字符
                #     if text_in3 == '01020101':
                #         self.checkBox_13.setChecked(True)
                #     else:
                #         self.checkBox_13.setChecked(False)
                #     print('text_IN3', text_in3)
                #     feedback_list.append(text_in3)
                # else:  #### 无返回数据
                #     no_feedback = modbus_rtu.writedata(self.ser,DO2_ON)  ###3号继电器打开   控制器无返回数据 D03 =1
                #     print('no_feedback data')
                #
                # if len(feedback_list) == 20:
                #     feedback_list.clear()
                # else:
                #     self.statistic_msg(str(feedback_list))

                #### 同步UI 信号
                # intput_box_list = [self.checkBox_10.isChecked(), self.checkBox_11.isChecked(), self.checkBox_12.isChecked(), self.checkBox_13.isChecked()]
                output_box_list =[self.checkBox_2.isChecked()]#,self.checkBox_3.isChecked(),self.checkBox_4.isChecked(),self.checkBox_5.isChecked()]

                for i , n in enumerate(output_box_list):
                    if n:
                        # print('scratch detected')
                        feedback_data = modbus_rtu.writedata(self.ser, DO3_ON)  ### OUT4 = 1
                        feedback_data = modbus_rtu.writedata(self.ser, DO2_OFF)  ###PLC控制，灭绿灯-240228
                    else:
                        # print('scratch has not detected')
                        feedback_data = modbus_rtu.writedata(self.ser, DO3_OFF)  ### OUT4 = 0
                        feedback_data = modbus_rtu.writedata(self.ser, DO2_ON)  ###PLC控制，亮绿灯-240228
                        time.sleep(0.1)
                        feedback_data = modbus_rtu.writedata(self.ser, DO2_OFF)
            else:
                modbus_flag = False
                print('modbus shut off')
                shut_coil = modbus_rtu.writedata(self.ser, DO_ALL_OFF)  ###OUT1-4  OFF  全部继电器关闭  初始化

                self.ser.close()



    def modbus_on_off(self):
        global modbus_flag
        # if not modbus_flag:
        if self.runButton_modbus.isChecked():
            print('runButton_modbus.isChecked')
            modbus_flag = True
            print('set  modbus_flag = True')
            try:
                self.ser, self.ret, error = modbus_rtu.openport(self.port_type, 9600, 5)  # 打开端口
            except Exception as e:
                print('openport erro -1', e)
                self.statistic_msg(str(e))

            if not self.ret:
                self.runButton_modbus.setChecked(False)
                self.runButton_modbus.setStyleSheet('background-color:rgb(220,0,0)') ### background = red
                MessageBox(
                    self.closeButton, title='Error', text='Connection Error: '+ str(error), time=2000,
                    auto=True).exec_()
                print('port did not open')
                try:
                    self.ser, self.ret, error = modbus_rtu.openport(self.port_type, 9600, 5)  # 打开端口
                    if self.ret:
                        _thread.start_new_thread(myWin.thread_mudbus_run, ())  # 启动检测 信号 循环
                except Exception as e:
                    print('openport erro-2', e)
                    self.statistic_msg(str(e))
            else: # self.ret is  True
                self.runButton_modbus.setChecked(True)
                _thread.start_new_thread(myWin.thread_mudbus_run, ())  # 启动检测 信号 循环
                self.runButton_modbus.setStyleSheet('background-color:rgb(0,0,0)')  ### background = red
        else: # shut down modbus
            print('runButton_modbus.is unChecked')
            modbus_flag = False
            self.runButton_modbus.setChecked(False)
            print('shut down modbus_flag = False')


    def stop(self):
        if not self.det_thread.jump_out:
            self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)


        # self.det_thread.stop()  #### bug-1 加入此语句 停止线程会卡死  未解决

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './auto_save/jpg'  ### save result as .mp4
        else:
            self.det_thread.save_fold = None

    def checkrate(self):  #####latency checkbox
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading rtsp stream', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self): #UI bottun 'cameraButton'
        try:
            self.stop()  # stop running thread
            print('chose_cam run')
            MessageBox(
                self.closeButton, title='Enumerate Cameras', text='Loading camera', time=2000, auto=True).exec_()# self.closeButton, title='Enumerate Cameras', text='Loading camera', time=2000, auto=True).exec_()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            print('enum_camera:', cams)
            self.statistic_msg('enum camera：{}'.format(cams))
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)
            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()  # choose source
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        print(msg)
        # self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()  #comboBox
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def change_device(self, x):
        self.device_type = self.comboBox_device.currentText()
        self.det_thread.device = self.device_type
        self.statistic_msg('Change device to %s' % x)

    def change_source(self, x): # while the comboBox_source has changed
        self.source_type = self.comboBox_source.currentText()
        self.det_thread.source = self.source_type
        self.statistic_msg('Change source to %s' % x)

    def change_port(self, x):
        self.port_type = self.comboBox_port.currentText()
        # self.det_thread.source = self.source_type
        self.statistic_msg('Change port to %s' % x)


    def open_file(self):
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def max_or_restore(self):  ### window size control
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()


    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):  ### input img_src  output to pyqt label
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):  ### predicttion  output
        global results
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0] ## append to List  while the value greater than 0
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]  ### reform the list
            # print('output result:', type(results), results)
            self.resultWidget.addItems(results)
            if len(results) :
                self.pushButton_okng.setText(f"NG :class= {len(results)}")
                self.pushButton_okng.setStyleSheet('''QPushButton{
                        font-size: 20px;
                        font-family: "Microsoft YaHei";
                        font-weight: bold;
                        border-radius: 4px;
                        background-color: rgb(240,20,30);
                        color: rgb(255, 255, 255);
                        }''')
                for i , n in enumerate(results):
                    # str = re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", i)
                    # print('class name = ', n)
                    if i == 0:
                        self.checkBox_2.setChecked(True)
                    # else:
                    #     self.checkBox_2.setChecked(False)
                    if i == 1:
                        self.checkBox_3.setChecked(True)
                    # else:
                    #     self.checkBox_3.setChecked(False)
                    if i == 2:
                        self.checkBox_4.setChecked(True)
                    # else:
                    #     self.checkBox_4.setChecked(False)
                    if i == 3:
                        self.checkBox_5.setChecked(True)
                    # else:
                    #     self.checkBox_5.setChecked(False)
                    # if i == 5:
                    #     self.checkBox_6.setChecked(True)

                    # self.checkBox_2.setText(str(i))
            else:
                self.pushButton_okng.setText(f"OK")
                self.pushButton_okng.setStyleSheet('''QPushButton{
                        font-size: 20px;
                        font-family: "Microsoft YaHei";
                        font-weight: bold;
                        border-radius: 4px;
                        background-color: rgb(0,220,127);
                        color: rgb(255, 255, 255);
                        }''')
                self.checkBox_2.setChecked(False)
                self.checkBox_3.setChecked(False)
                self.checkBox_4.setChecked(False)
                self.checkBox_5.setChecked(False)
                self.checkBox_6.setChecked(False)
                # self.checkBox_2.setText("")
                # print("result = []")

        except Exception as e:
            print(repr(e))

    def load_setting(self):
        config_file = 'config/setting.json'

        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            device = 0
            port = "COM3"
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck,
                          "device": device,
                          "port": port
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            print('load config:',type(config), config)
            if len(config) != 8 : ### 参数不足时  补充参数
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
                device = 0
                port = "COM3"
                source = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
                device = config['device'] ## index number
                port = config['port'] ## index number
                source = config['source']
        ### 依据存储的json文件 更新 ui参数
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save() ###auto save  checkbox

        self.comboBox_device.setCurrentIndex(device) # 设置当前索引号 "device": 0
        self.comboBox_port.setCurrentIndex(port)  # 设置当前索引号 "port": "COM0"
        self.comboBox_source.setCurrentIndex(source)  # 设置当前索引号 "port": "COM0"
    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_path = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()  ### Latency funtion
        config['savecheck'] = self.saveCheckBox.checkState() ### Auto Save
        config['device'] = self.comboBox_device.currentIndex() ### 获取当前索引号
        config['port'] = self.comboBox_port.currentIndex()  ### 获取当前索引号
        config['source'] = self.comboBox_source.currentIndex()  ### 获取当前索引号
        ####新增参数 请在此处添加， 运行UI后 点击关闭按钮 后保存为 json文件 地址= ./config/setting.json
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_json)
            print('confi_json write')
        MessageBox(
            self.closeButton, title='Tips', text='Program is exiting.', time=2000, auto=True).exec_()
        sys.exit(0)

    def load_config(self):   ####  初始化 modbus connection
      global winsound_freq, winsound_time, winsound_freq_2, winsound_time_2
      try:
          ### 提取备份数据 当出现断电关机数据丢失时， 将Cahce中 备份文件拷贝出来
          cache_path = os.path.dirname(os.path.realpath(__file__)) + r'\config'
          to_path = os.path.dirname(os.path.realpath(__file__))  ### root path

          for root, dirs, files in os.walk(
                  cache_path):  # root 表示当前正在访问的文件夹路径# dirs 表示该文件夹下的子目录名list # files 表示该文件夹下的文件list
              # print('files',files) ####['edgevalue.db.bak', 'edgevalue.db.dat', 'edgevalue.db.dir']
              for i in files:
                  from_path = os.path.join(root, i)  # 合并成一个完整路径
                  # copy(from_path, to_path)  ### 第一个参数 是复制对象， 第二个是 复制到文件夹
                  # print('from_path', from_path)
                  # print('to_path', to_path)
              print('files in config has been coppied sucessfully')

          # self.ser, self.ret , error = modbus_rtu.openport(self.port_type, 9600, 5)  # 打开端口

      except Exception as e:
          print('openport erro', e)
          self.statistic_msg(str(e))




####  for  testing  ↓ ##################################################
def cvshow_image(img):  ### input img_src  output to pyqt label
    try:
        cv2.imshow('Image', img)
    except Exception as e:
        print(repr(e))


if __name__ == "__main__":

    app = QApplication(sys.argv)
    myWin = MainWindow() #### 实例化
    myWin.show()
    print('prameters load completed')
    myWin.runButton_modbus.setChecked(True)
    myWin.modbus_on_off()### start modbus
    # time.sleep(1)
    # print('thread_mudbus_run start')
    # _thread.start_new_thread(myWin.thread_mudbus_run, ())  #### 启动检测 信号 循环


    #### 调试用代码
    # det_thread = DetThread() #### 实例化
    # det_thread.weights = "pt/yolov5s.pt"
    # det_thread.device = '0'
    # det_thread.source = 'streams.txt'
    # det_thread.is_continue = True
    # det_thread.start()   ###
    # # ##### connect UI  调试输出到 UI  ↓
    # det_thread.send_img_ch0.connect(lambda x: myWin.show_image(x, myWin.video_label_ch0))
    # det_thread.send_img_ch1.connect(lambda x: myWin.show_image(x, myWin.video_label_ch1))
    # det_thread.send_img_ch2.connect(lambda x: myWin.show_image(x, myWin.video_label_ch2))
    # det_thread.send_img_ch3.connect(lambda x: myWin.show_image(x, myWin.video_label_ch3))

    # 单独输出 调试模式 ↓
    # det_thread.send_img_ch0.connect(lambda x: cvshow_image(x))

    # myWin.showMaximized()
    sys.exit(app.exec_())
