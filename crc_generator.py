def calculate_crc(data):
    # 将字符串转换为十六进制数组
    data_array = [int(x, 16) for x in data.split(' ')]

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
    crc_code = data + ' ' + format(crc & 0xFF, '02X') + ' ' + format((crc >> 8) & 0xFF, '02X')

    return crc_code

raw_data = [1, 6, 1, 100]

# 将 raw_data 转换为 hex_data
hex_data = [format(x, 'X').zfill(4) for x in raw_data]
string = hex_data[0]
hex_data[0] = string[-2:]
string = hex_data[1]
hex_data[1] = string[-2:]

# 将 hex_data 转换为 str_data
str_data = ' '.join([x[i:i+2] for x in hex_data for i in range(0, len(x), 2)])

print("raw_data:", raw_data)
print("hex_data:", hex_data)
print("str_data:", str_data)


hex_data: ['0001', '0006', '0001', '0064']
hex_data: ['01', '06', '0001', '0064']
# 测试
# raw_data = [1, 5, 13066, 00]
# hex_data = ['{:02X}'.format(x) for x in raw_data]
# input_data = ' '.join(hex_data)
# print("raw_data =", raw_data)
# print("hex_data =", hex_data)
# print("input_data =", input_data)
# crc_code = calculate_crc(input_data)
# print(crc_code)
#
# #
# if __name__ == "__main__":
#
# data = [0x01, 0x04, 0x03, 0x26, 0x00, 0x01]
# crc_code = generate_modbus_crc(data)
# print("Modbus CRC校验码为：", crc_code)