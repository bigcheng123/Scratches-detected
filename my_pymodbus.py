
'''
#MOUDBUS TCP
from pymodbus.client.sync import ModbusTcpClient

# 创建Modbus TCP客户端对象并连接到PLC
plc_ip = '192.168.0.1'  # PLC IP地址
port = 502  # Modbus TCP默认端口号为502
client = ModbusTcpClient(plc_ip, port)
connection = client.connect()
if not connection:
    print("无法连接到PLC")
else:
    try:
        # 从指定的起始地址开始读取4个寄存器（32位）的值
        start_address = 0x0000  # 起始地址
        num_registers = 4  # 寄存器数量
        response = client.read_holding_registers(start_address, count=num_registers, unit=1)

        if response.isError():
            print('读取错误')
        else:
            registers_values = [value for value in response.registers]

            # 打印每个寄存器的值
            for i, register_value in enumerate(registers_values):
                print(f"Register {i + 1}: {hex(register_value)} ({register_value})")

    except Exception as e:
        print(f"发生错误：{str(e)}")
finally:
    client.close()
请确保你已安装pymodbus库，并替换示例代码中的IP地址、端口号、寄存器地址和寄存器数量等参数为你实际的设备信息
'''


'''MOUDBUS RTU '''
# from pymodbus.client.sync import ModbusSerialClient as ModbusClient
# # from pymodbus.client.sync_client import ModbusSerialClient as ModbusClient
#
# # 串口名称，根据你的实际情况修改
# serial_port = "COM10"
#
# # 创建Modbus客户端对象
# client = ModbusClient(method='rtu', port=serial_port)
#
# try:
#     # 打开与PLC的通信
#     client.connect()
#
#     # 从指定地址读取1个寄存器的值
#     # 根据实际情况修改起始地址、计数和单元标识符
#     result = client.read_holding_registers(address=0x0000, count=1, unit=1)
#
#     if not isinstance(result, Exception):
#         print("成功读取到寄存器数据：", result.registers[0])
#
# except Exception as e:
#     print("发生错误：", str(e))
# finally:
#     # 关闭与PLC的通信
#     client.close()
#

from pymodbus.client.sync import ModbusSerialClient as ModbusClient

# 串口名称，根据你的实际情况修改
serial_port = "COM10"

# 创建Modbus客户端对象
client = ModbusClient(method='rtu', port=serial_port, baudrate=9600, stopbits=1, bytesize=8, parity='N')

try:
    # 打开与PLC的通信
    client.connect()

    # 写入线圈（如果M1是线圈类型的话，这通常不是必须的，除非你需要设置它）
    # 注意：线圈地址和值需要根据实际情况调整
    result_write = client.write_coil(address=0x0004, value=True)

    # 读取线圈状态（包括M1寄存器的值）
    # 从地址0x0001开始，读取1个线圈的状态
    result_read = client.read_coils(address=0x0004, count=1)

    if not isinstance(result_read, Exception):
        # 输出M1寄存器的值
        print("M1 寄存器的值:", result_read.bits[0])

    if not isinstance(result_write, Exception):
        print("线圈写入成功")

except Exception as e:
    print("发生错误：", str(e))
finally:
    # 关闭与PLC的通信


# from pymodbus.client.sync import ModbusSerialClient as ModbusClient
# import logging
# logging.basicConfig()
# log = logging.getLogger()
# log.setLevel(logging.DEBUG)
#
# UNIT = 0x1
# def run_sync_client():
#     client = ModbusClient(method='rtu', port='com10', baudrate=9600, timeout=1)  # 客户机(通信方式，端口，波特率，超时)
#     client.connect()
#
#     # log.debug("读保持寄存器，返回成功与否")
#     # rr = client.read_holding_registers(1, 1, unit=UNIT)  # 03H读保持寄存器(起始寄存器号，数量，从机号)->返回成功与否
#     # print(rr)
#     # client.close()
#
#     log.debug("写保持寄存器并读回")
#     rq = client.write_register(0, 333, unit=UNIT)  # 06H写保持寄存器(起始寄存器号，值，从机号)->返回写的数值
#     print(rq)  # 写入的数值
#     print(rq.function_code)  # 功能码
#     rr = client.read_holding_registers(0, 8, unit=UNIT)  # 03H读保持寄存器(起始寄存器号，数量，从机号)->返回成功与否
#     print(rr)
#     print(rr.registers)  # 读出的数据列表
#     assert (rq.function_code < 0x80)  # test that we are not an error
#     assert (rr.registers[1] == 666)  # test the expected value
#
#
# if __name__ == "__main__":
#     run_sync_client()

