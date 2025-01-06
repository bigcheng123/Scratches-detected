# -*- coding:utf-8 -*-
# Author:
# Windows7&Python3.7

import serial  ###pip install pyserial
import time
import cv2
import threading
import random

STRGLO =  "" #读取的数据
serflag = True  #读取标志位
#读数代码本体实现
def ReadData(ser):
    global STRGLO , serflag
    # 循环接收数据，此为死循环，可用线程实现
    while serflag:
        if ser.in_waiting:
            STRGLO = ser.read(ser.in_waiting).hex() ###读取到的数据是 二进制  转换位hex() 16进制
            # print(STRGLO)
#打开串口
# 端口，GNU / Linux上的/ dev / ttyUSB0 等 或 Windows上的 COM3 等
# 波特率，标准值之一：50,75,110,134,150,200,300,600,1200,1800,2400,4800,9600,19200,38400,57600,115200
# 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）

def openport(port,baudrate,timeout):  ### 降低PORT口的 数据缓存设定 可以提高通信质量
    ret = False
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)   # 选择串口，并设置波特率
        if not ser.is_open:
            ser.open()
            # threading.Thread(target=ReadData, args=(ser,)).start()
    except Exception as error:
        print("---OPEN PORT ERROR---：", error)
        ser = None
        ret = False
        return ser, ret, error
    else:
        ret = True
        error = "PORT SUCCESSFULLY"
        return ser, ret, error

#关闭串口
def DColsePort(ser):
    global serflag
    serflag = False
    ser.close()

#写数据
def DWritePort(ser,text):
    result = ser.write(bytes.fromhex(text))  # 写数据  使用 HEX TO  bytes    16进制 转 二进制
    return result   #### 返回代码

#读数据
def DReadPort():
    global STRGLO
    str = STRGLO
    STRGLO = ""  #清空 当次  读取
    return str

    # serial .isOpen()

def writedata(ser, hexcode):
    # lock= threading.Lock()
    # openport()
    # if (ser.is_open):
    # lock.acquire()
    str_return_data = ""
# try:
    # time.sleep(0.2)
    # time.sleep(random.random()*0.5) # 每次write port要 分隔开  防止同时进行 写操作 间隔越大
    send_data = bytes.fromhex(hexcode)    # HEX码 转换 bytes 字节码     发送数据转换为b'\xff\x01\x00U\x00\x00V'
    ser.write(send_data)   # 发送命令
    time.sleep(0.02)        # 延时，否则len_return_data将返回0，此处易忽视！！！ 延迟低于 0.01无法接收数据
    len_return_data = ser.inWaiting()  # 获取缓冲数据（接收数据）长度
    # if len_return_data:
    return_data = ser.read(len_return_data)  # 读取缓冲数据
    # bytes(2进制)转换为hex(16进制)，应注意Python3.7与Python2.7此处转换的不同，并转为字符串后截取所需数据字段，再转为10进制
    str_return_data = str(return_data.hex())   # bytes(2进制)转换为hex(16进制
    # feedback_data = int(str_return_data[-6:-2], 16) ### j截取所需字段
    # feedback_data = int(str_return_data[-6:-2], 16)
    # print(feedback_data)
    # print(str_return_data)
# except Exception as e:
#     print('writedata error',e)       
# else:
#     pass
#     global str_return_data
    return(str_return_data)  ##返回数据

    # lock.release()
# else:
    #     print("open failed")


# def closeport(ser):
#     ser.colse()
#     print("port close success")
def str2bool(feedback_data):
    # hexcode = 'FE 02 00 01 00 01 FC 05' ### READ IN2 
    # feedback_data = openport(hexcode)
    # print('feedback_data',feedback_data)
    BOOL =  False
    try:
        time.sleep(random.random()*0.5) #加入随机延迟 错开信号
        if feedback_data:
            str_result = int(feedback_data[6:8], 10)  ###取目标字符   ##返回的代码已经时10进制 010201016048
            # print('feedback_data',str_result)
            if str_result == 1:
                BOOL = True
            else:
                BOOL = False 
            # ser.close()  ### 关闭串口
            # print("port close success")  
    except Exception as e:
        print('readdata error',e)  
    else:
        pass

    return (BOOL)

def calculate_crc(raw_data):  #计算CRC校验码
        # 参数raw_data 全部采用十进制格式列表： [站号 , 功能码, 软元件地址 , 读写位数, 数据] 示例raw_data = [1, 6, 10, 2, 111]
        # 将 raw_data（DEC格式） 转换为 hex_data
        hex_data = [format(x, 'X').zfill(4) for x in raw_data]
        string = hex_data[0]
        hex_data[0] = string[-2:]
        string = hex_data[1]
        hex_data[1] = string[-2:]
        try:
            hex_data[4]  # 确认是否有第五位→多寄存器读写
        except:
            print("no hex 4")
        else:
            string = hex_data[4]
            hex_data[4] = string[-2:]  # 多寄存器位数信息
            string = hex_data[5]
            hex_data[5] = string.zfill(8)  # 写入2个寄存器/补充至8位16进制数
            string = hex_data[5]
            string1 = string[-4:]  # 串口写入高低位与PLC高低位逻辑不一致，需要高低4位互换
            string2 = string[:4]
            hex_data[5] = ''.join([string1, string2])
        # 将 hex_data 转换为 str_data
        str_data = ' '.join([x[i:i + 2] for x in hex_data for i in range(0, len(x), 2)])
        # 将字符串转换为十六进制数组
        data_array = [int(x, 16) for x in str_data.split(' ')]

        # 计算CRC校验码
        crc = 0xFFFF
        for i in range(len(data_array)):
            crc ^= data_array[i]
            for j in range(8):
                if crc & 1:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        # 将CRC校验码添加到原始数据后面
        crc_code = str_data + ' ' + format(crc & 0xFF, '02X') + ' ' + format((crc >> 8) & 0xFF, '02X')
        return crc_code  # return str  crc_data: 01 06 00 0A 00 6F E9 E4


if __name__ == '__main__':
    #### hexcode  ######
    IN0_READ = '01 02 00 00 00 01 B9 CA'
    IN1_READ = '01 02 00 01 00 01 E8 0A'
    IN2_READ = '01 02 00 02 00 01 18 0A'
    IN3_READ = '01 02 00 03 00 01 49 CA'
    DO0_ON = '01 05 00 00 FF 00 8C 3A'
    DO0_OFF = '01 05 00 00 00 00 CD CA'
    DO1_ON = '01 05 00 01 FF 00 DD FA'
    DO1_OFF = '01 05 00 01 00 00 9C 0A'
    DO2_ON = '01 05 00 02 FF 00 2D FA'
    DO2_OFF = '01 05 00 02 00 00 6C 0A'
    DO3_ON = '01 05 00 03 FF 00 7C 3A'
    DO3_OFF = '01 05 00 03 00 00 3D CA'

    DO_ALL_ON = '01 0F 00 00 00 04 01 FF 7E D6'
    DO_ALL_OFF = '01 0F 00 00 00 04 01 00 3E 96'  ##OUT1-4  OFF  全部继电器关闭  初始化



    # read_in1()
    ser, ret, _ = openport(port='COM51', baudrate=9600, timeout=5) #打开端口port,baudrate,timeout
    n=10
    str_result=''

    while  n:
        # t = threading.Thread(target= writedata,args=(ser,'01 02 00 00 00 01 B9 CA'))
        # print(t)
        start = time.time()
        # writ coil  [站号 , 功能码, 软元件地址 , 读写位数, 数据]
        # write_m2_on = '01 05 00 02 FF 00 2D FA '#calculate_crc([1, 5, 2, 65280])
        # writedata(ser, '01 05 00 01 FF 00 DD FA')
        # print('write_m2_on', calculate_crc([1, 5, 2, 65280]))
        # write_coil = writedata(ser, calculate_crc([1, 5, 3, 65280]))  # 程序运行后闭合线圈M10
        # # write registers
        time.sleep(0.2)
        # writeD10 = calculate_crc([1, 16, 10, 2, 4, n])  ## 批量寄存器写入 D10 + D11 (2表示写入2位）
        # print(f'writeD10 {writeD10}')
        # write_register = writedata(ser, writeD10)  # 向 PLC NG计数 D10 写入
        # writeD1 = calculate_crc([1, 16, 1, 2, 4, 22222])  ## 批量寄存器写入 D10 + D11 (2表示写入2位）
        # print('writeD1', writeD1)
        # writedata(ser, writeD1)  #
        stop = time.time()
        speed = (stop - start) * 1000 # 转化为 毫秒单位/ms
        freq = 1000 / speed  ## 每秒钟循环次数
        print(f'speed: {speed}ms', f'freq= {freq}')
        n = n-1

    ser.close()
    print(('sel.close'))



####    功能码  IN 读取操作
# FE 02 00 00 00 04 6D C6 ## 读取4路IN
# FE 02 01 01 50 5C ##  IN1_ON
# FE 02 01 02 10 5D ##  IN2_ON
# FE 02 01 04 90 5F ##  IN3_ON
# FE 02 01 08 90 5A ##  IN4_ON


# FE 02 00 00 00 01 AD C5  ## READ IN1
# FE 02 01 00 91 9C  ## IN1_OFF
# FE 02 01 01 50 5C  ## IN1_ON

# FE 02 00 01 00 01 FC 05 ### READ IN2
# FE 02 01 00 91 9C  ### IN2_OFF
# FE 02 01 01 50 5C  ### IN2_ON

# FE 02 00 02 00 01 0C 05  ### READ IN3
# FE 02 01 00 91 9C ## IN3_OFF
# FE 02 01 01 50 5C  ## IN3_ON

# FE 02 00 03 00 01 5D C5 ## READ IN4
# FE 02 01 00 91 9C ## IN4_OFF
# FE 02 01 01 50 5C  ## IN4_ON



# #####  线圈写操作
# 01 0F 00 00 00 04 01 00 3E 96  # 全部COIL关闭
# #返回码 01 0F 00 00 00 04 54 08 
# 01 0F 00 00 00 04 01 FF 7E D6  # 全部COIL打开
# #返回码 01 0F 00 00 00 04 54 08
