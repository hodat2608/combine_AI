from glob import glob     
import threading

path1 = 'C:\Users\CCSX009\Desktop\FH\CD\CHAU1\**\*.jpg'
path2 = 'C:\Users\CCSX009\Desktop\FH\CD\CHAU2\**\*.jpg'
path3 = 'C:\Users\CCSX009\Desktop\FH\TC\CHOI1\**\*.jpg'
path4 = 'C:\Users\CCSX009\Desktop\FH\TC\CHOI2\**\*.jpg'

def loop1():
    for f1 in path1:
        cd =1
        program_camera2_FH(model=model2,size=416,conf=values['conf_thres2']/100, file = f1)   
def loop2():
    for f1 in path2:
        cd =1
        program_camera2_FH(model=model2,size=416,conf=values['conf_thres2']/100, file = f1)   
def loop3():
    for f1 in path3:
        cd =1
        program_camera3_FH(model=model2,size=416,conf=values['conf_thres2']/100, file = f1)   
def loop4():
    for f1 in path4:
        cd =1
        program_camera3_FH(model=model2,size=416,conf=values['conf_thres2']/100, file = f1)
thread1 = threading.Thread(target=loop1).start()
thread2 = threading.Thread(target=loop2).start()
thread3 = threading.Thread(target=loop3).start()
thread4 = threading.Thread(target=loop4).start()
thread1.join()
thread2.join()
thread3.join()
thread4.join()