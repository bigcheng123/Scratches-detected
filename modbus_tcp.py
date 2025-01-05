import time

from pymodbus.client import ModbusTcpClient


def modbustcp_open_port(ip, port):
    global client
    try:
        # 创建 Modbus TCP 客户端
        client = ModbusTcpClient(ip, port)
        # 连接到服务器
        connection = client.connect()

        if not connection:
            print("无法连接到 Modbus 服务器。")
            return client
    except Exception as e:
        print(f"连接发生错误：{e}")
        client = None
        return client


def modbustcp_write_coil(address, write_value):

    try:
        # 写入线圈
        result = client.write_coil(address, write_value)

        if result.isError():
            print(f"写入失败：{result}")
        else:
            print(f"成功写入线圈 {address}，值：{value}")
    except Exception as e:
        print(f"发生错误：{e}")
    # finally:
    #     # 关闭连接
    #     connect_plc.client.close()

def modbustcp_write_register(address, decimal_value):
    """
    写入一个32位整数到两个寄存器。 一个地址的存储范围 -32768 ~ 32767
    :param address: 起始寄存器地址
    :param decimal_value: 要写入的32位整数
    """
    # 确保输入值在32位范围内（0~2^32-1）
    if not (0 <= decimal_value < 2 ** 32):
        raise ValueError("Input value must be a 32-bit unsigned integer")

    # 获取低16位和高16位
    low_16_bits = decimal_value & 0xFFFF  # 低16位
    high_16_bits = (decimal_value >> 16) & 0xFFFF  # 高16位

    # 将两个16位数合并成列表
    write_values = [low_16_bits, high_16_bits]
    try:
        # 写入两个寄存器
        result = client.write_registers(address, write_values)

        if result.isError():
            print(f"写入失败：{result}")
        else:
            print(f"成功写入寄存器 {address} 和 {address + 1}，值：{write_values}")
    except Exception as e:
        print(f"发生错误：{e}")


def modbustcp_read_register(address, count):
    """
    读取指定数量的寄存器并返回处理后的数据
    :param address: 起始地址
    :param count: 要读取的寄存器数量
    :return: 如果读取的是两个寄存器，则返回处理后的 32 位整数值；否则返回寄存器值列表
    """
    try:
        # 读取寄存器
        response = client.read_holding_registers(address, count)
        if response.isError():
            print(f"读取失败：{response}")
            return None
        registers = response.registers

        # 如果读取了两个寄存器，重新组合为 32 位整数
        if count == 2:
            print('读取2个寄存器 将直接返回处理后的数据')
            low_16_bits, high_16_bits = registers[0], registers[1]
            reconstructed_value = (high_16_bits << 16) | low_16_bits
            return reconstructed_value

        # 如果读取的寄存器数量不是 2，直接返回原始值
        print(f'返回原始数据:{registers}')
        return registers
    except Exception as e:
        print(f"读取发生错误：{e}")
        return None


def modbustcp_client_close():
    try:
        result = client.close()
        if not result:
            print("成功关闭通讯")
        else:
            print(f"无法关闭modbus。")

    except Exception as e:
        print(f"连接发生错误：{e}")



if __name__ == "__main__":
    # 配置 Modbus 服务器地址和端口
    modbus_ip = "192.168.0.111"  # 替换为您的 Modbus 服务器 IP
    modbus_port = 502            # 默认 Modbus TCP 端口

    # 配置要写入的线圈地址和值
    coil_address = 0  # 线圈地址
    coil_value = True  # 要写入的值（True 或 False）

    register_address = 2
    value = 988881111
    # register_1, register_2 = split_into_registers(decimal_value)
    # register_value = [register_1, register_2]
    modbustcp_open_port(modbus_ip, modbus_port)  # open port
    n=10
    while n:
        # 写入线圈
        start = time.time()
        modbustcp_write_coil(coil_address, coil_value)  # write coil
        modbustcp_write_register(register_address, value)  # write register
        result = modbustcp_read_register(register_address, count=2)
        print(f'holding_registers: {result}')

        stop = time.time()
        speed = (stop - start) * 1000 # 转化为 毫秒单位/ms
        freq = 1000 / speed  ## 每秒钟循环次数
        print(f'speed: {speed}ms', f'freq= {freq}')
        n= n-1

    modbustcp_client_close()  # close port