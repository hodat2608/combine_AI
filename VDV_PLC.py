import socket
import time

#frame = {'data': [2,0xff,0x0a,0,0x32,0,0,0,0x20,0x4d,0x02,0,0x00]}
#def send_frame(frame):
#    dummy =[]
#    for field in list(frame.values()):
#        dummy += field
#    return dummy

#"Doc bit. Ex: M10 => name: 'M', number_name: 100"
def read_bit(name, number_name):
    frame = {'data': [0,0xff,0x0A,0,number_name,0,0,0,0x20,(ord(name)),1,0]}
    dummy =[]
    for field in list(frame.values()):
        dummy += field
    soc.sendall(bytes(dummy))
    data = soc.recv(1024)
    #print(data)
    datalist = list(data)
    #print (datalist)
    if datalist[2] == 0:
        return 0
    else:
        return 1
    
#"Ghi bit. Ex: M100 ON => name: 'M', number_name: 100, status: 1=ON - 0=OFF"
def write_bit(name, number_name, status):
    if status == 1:
        bit = 16
    else:
        bit = 0
    frame = {'data': [2,0xff,0x0A,0,number_name,0,0,0,0x20,(ord(name)),1,0,bit]}
    dummy =[]
    for field in list(frame.values()):
        dummy += field
    soc.sendall(bytes(dummy))
    data = soc.recv(1024)
    #print (data)
    datalist = list(data)
    #print (datalist)
    return True

#"Doc thanh ghi"
def read_word(name, number_name):
    frame = {'data': [1,0xff,0x0A,0,number_name,0,0,0,0x20,(ord(name)),1,0]}
    dummy =[]
    for field in list(frame.values()):
        dummy += field
    soc.sendall(bytes(dummy))
    data = soc.recv(1024)
    datalist = list(data)
#     print (datalist)
    doiso1 = format(datalist[2], "b")
    str1 = '00000000'
    doiso2 = format(datalist[3], "b")
    chuoibinary=doiso2+str1[0:(8-len(doiso1))]+doiso1
#     print(chuoibinary)
    if int(chuoibinary, 2) > 32768:
        return (int(chuoibinary, 2) - 2**16)
    else:
        return (int(chuoibinary, 2))

#'Ghi thanh ghi'
def write_word(name, number_name, data):
    a = data
    if 15 >= a >= 0:
        str1 = format(a,'b')
        doiso1 = str1[:]
        doiso2 = '00000000'
    elif 2**8 > a > 15:
        str1 = format(a,'b')
        doiso1 = str1[:]
        doiso2 = '00000000'
    elif a >= 2**8:
        str1 = format(a,'b')
        doiso1 = str1[(len(str1)-8):]
        doiso2 = str1[0:(len(str1)-8)]
    else:
        b = a + 2**16
        str2 = format(b,'b')
        doiso1 = str2[(len(str2)-8):]
        doiso2 = str2[0:(len(str2)-8)]
    frame = {'data1': [3,0xff,0x0A,0,number_name,0,0,0,0x20,(ord(name)),
                           1,0,int(doiso1,2),int(doiso2,2)]}
    dummy =[]
    for field in list(frame.values()):
        dummy += field
        soc.sendall(bytes(dummy))
    data = soc.recv(1024)
#    print (data)
#    datalist = list(data)
    return True

#"Khoi tao socket"
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#"Ket noi voi PLC"
def socket_connect(host, port):
    try:
        soc.connect((host, port))
        return True
    except OSError:
        print("Can't connect to PLC")
        time.sleep(3)
        print("Reconnecting....")
        return False

#connect
if __name__ == '__main__':
    connected = False
    while connected == False:
        connected = socket_connect('192.168.1.250', 8000)     
    print("connected")