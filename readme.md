### TIPS

This repo is based on [YOLOv5 v6.1](https://github.com/ultralytics/yolov5/tree/v6.1)

This repo is including tube defect model in /pt folder.<br>
You can also download the models of  YOLOv5 v6.1 from [here](https://github.com/ultralytics/yolov5/releases/tag/v6.1)，and put the them to the /pt folder. When the GUI runs, the existing models will be automatically detected.

PLC program based on Mitsubishi FX3S with MODBUS module

HIM program based on Weinview MT8072IP



### Demo Video：
[Demo:https://www.bilibili.com/video/BV1nz421S7KR](https://www.bilibili.com/video/BV1nz421S7KR)

### Install
You need to install [conda](https://www.anaconda.com/download/success) first,and then you need to prepare torch.whl==1.8.1 and torchvision.whl==0.9.1 and put them to the /install_torch folder<br>
You can download .whl file from [tsinghua mirror station](https://mirrors.tuna.tsinghua.edu.cn/)

```bash
git clone https://gitee.com/trgtokai/scratch-detect.git
cd scratch-detect
conda create -n yolov5_pyqt5 python=3.8
conda activate yolov5_pyqt5
pip install -r requirement-trg.txt
python main.py
```

### Function

1. support image/video/multicamera/rtsp as input
2. change model
3. change IoU
4. change confidence
5. set latency
6. paly/pause/stop
7. result statistics
8. save  detected image/video automatically
9. use ModbusRTU to communicate with PLC 

You can find ui files in [main_win](./main_win) and [dialog](dialog)

### About Packaging

- install pyinstaller

```
pip install pyinstaller==5.7.0
```

- package the GUI

```
pyinstaller -D -w --add-data="./utils/*;./utils" --add-data="./config/*;./config" --add-data="./icon/*;./icon" --add-data="./pt/*;./pt" --add-data="./imgs/*;./imgs" main.py
```

- if no errors occur, the packaged application is in dist/main
