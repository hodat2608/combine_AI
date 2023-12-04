from glob import glob
from tkinter.tix import Tree
import os, cv2, torch, time, datetime, shutil
#import threading
import numpy as np 
import pandas as pd
import PySimpleGUI as sg
from PIL import Image, ImageTk
import connect_PLC_Mitsubishi as plc
import traceback
import sqlite3
import keyboard

SCALE_X_CAM1 = 640*0.8/2048
SCALE_Y_CAM1 = 480*0.8/1536

SCALE_X_CAM2 = 640/1440
SCALE_Y_CAM2 = 480/1080

def check_image(image_path):
    try:
        image = Image.open(image_path)
        image.verify()
        a = "khongloi"
    except (IOError, SyntaxError) as e:
        a = "loi"
    return a

def removefile():
    directory1 = 'C:/FH/CAM1/**/*'
    directory2 = 'C:/FH/CAM2/**/*'
    chk1 = glob(directory1)
    for f1 in chk1:
        fname1=os.path.dirname(f1)
        shutil.rmtree(fname1)
    chk2 = glob(directory2)
    for f2 in chk2:
        fname2=os.path.dirname(f2)
        shutil.rmtree(fname2)
    print('already delete folder')

'''
#Dung cho camera truc tiep
class CMyCallback:
    """
    Class that contains a callback function.
    """

    def __init__(self):
        self._image = None
        self._lock = threading.Lock()

    @property
    def image(self):
        """Property: return PyIStImage of the grabbed image."""
        duplicate = None
        self._lock.acquire()
        if self._image is not None:
            duplicate = self._image.copy()
        self._lock.release()
        return duplicate

    def datastream_callback1(self, handle=None, context=None):
        """
        Callback to handle events from DataStream.

        :param handle: handle that `trigger` the callback.
        :param context: user data passed on during callback registration.
        """
        st_datastream = handle.module
        if st_datastream:
            with st_datastream.retrieve_buffer() as st_buffer:
                # Check if the acquired data contains image data.
                if st_buffer.info.is_image_present:
                    # Create an image object.
                    st_image = st_buffer.get_image()

                    # Check the pixelformat of the input image.
                    pixel_format = st_image.pixel_format
                    pixel_format_info = st.get_pixel_format_info(pixel_format)

                    # Only mono or bayer is processed.
                    if not(pixel_format_info.is_mono or pixel_format_info.is_bayer):
                        return

                    # Get raw image data.
                    data = st_image.get_image_data()

                    # Perform pixel value scaling if each pixel component is
                    # larger than 8bit. Example: 10bit Bayer/Mono, 12bit, etc.
                    if pixel_format_info.each_component_total_bit_count > 8:
                        nparr = np.frombuffer(data, np.uint16)
                        division = pow(2, pixel_format_info
                                       .each_component_valid_bit_count - 8)
                        nparr = (nparr / division).astype('uint8')
                    else:
                        nparr = np.frombuffer(data, np.uint8)

                    # Process image for display.
                    nparr = nparr.reshape(st_image.height, st_image.width, 1)

                    # Perform color conversion for Bayer.
                    if pixel_format_info.is_bayer:
                        bayer_type = pixel_format_info.get_pixel_color_filter()
                        if bayer_type == st.EStPixelColorFilter.BayerRG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_RG2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGR:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GR2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGB:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GB2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerBG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_BG2RGB)

                    # Resize image and store to self._image.
                    nparr = cv2.resize(nparr, None,
                                       fx=SCALE_X_CAM1,
                                       fy=SCALE_Y_CAM1)
                    self._lock.acquire()
                    self._image = nparr
                    self._lock.release()


    def datastream_callback2(self, handle=None, context=None):
        """
        Callback to handle events from DataStream.

        :param handle: handle that trigger the callback.
        :param context: user data passed on during callback registration.
        """
        st_datastream = handle.module
        if st_datastream:
            with st_datastream.retrieve_buffer() as st_buffer:
                # Check if the acquired data contains image data.
                if st_buffer.info.is_image_present:
                    # Create an image object.
                    st_image = st_buffer.get_image()

                    # Check the pixelformat of the input image.
                    pixel_format = st_image.pixel_format
                    pixel_format_info = st.get_pixel_format_info(pixel_format)

                    # Only mono or bayer is processed.
                    if not(pixel_format_info.is_mono or pixel_format_info.is_bayer):
                        return

                    # Get raw image data.
                    data = st_image.get_image_data()

                    # Perform pixel value scaling if each pixel component is
                    # larger than 8bit. Example: 10bit Bayer/Mono, 12bit, etc.
                    if pixel_format_info.each_component_total_bit_count > 8:
                        nparr = np.frombuffer(data, np.uint16)
                        division = pow(2, pixel_format_info
                                       .each_component_valid_bit_count - 8)
                        nparr = (nparr / division).astype('uint8')
                    else:
                        nparr = np.frombuffer(data, np.uint8)

                    # Process image for display.
                    nparr = nparr.reshape(st_image.height, st_image.width, 1)

                    # Perform color conversion for Bayer.
                    if pixel_format_info.is_bayer:
                        bayer_type = pixel_format_info.get_pixel_color_filter()
                        if bayer_type == st.EStPixelColorFilter.BayerRG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_RG2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGR:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GR2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGB:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GB2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerBG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_BG2RGB)

                    # Resize image and store to self._image.
                    nparr = cv2.resize(nparr, None,
                                       fx=SCALE_X_CAM2,
                                       fy=SCALE_Y_CAM2)
                    self._lock.acquire()
                    self._image = nparr
                    self._lock.release()


def set_enumeration(nodemap, enum_name, entry_name):
    enum_node = st.PyIEnumeration(nodemap.get_node(enum_name))
    entry_node = st.PyIEnumEntry(enum_node[entry_name])
    enum_node.set_entry_value(entry_node)

def setup_camera1_stc():
    #lobal error_cam1
    #while error_cam1 == True:
    try:
        st_device1 = st_system.create_first_device()
        print('Device1=', st_device1.info.display_name)
        st_datastream1 = st_device1.create_datastream()
        callback1 = st_datastream1.register_callback(cb_func1)
        st_datastream1.start_acquisition()
        st_device1.acquisition_start()
        remote_nodemap1 = st_device1.remote_port.nodemap
        set_enumeration(remote_nodemap1,"TriggerMode", "Off")
        error_cam1 = False
        return  st_datastream1, st_device1,remote_nodemap1

    except Exception as exception:
        print(' Error Cam 1:', exception)
        str_error = "Error"
        window['result_cam1'].update(value= str_error, text_color='red',)

def setup_camera2_stc():
    #global error_cam2
    #while error_cam2 == True:
    try:
        st_device2 = st_system.create_first_device()
        print('Device2=', st_device2.info.display_name)
        st_datastream2 = st_device2.create_datastream()
        callback2 = st_datastream2.register_callback(cb_func2)
        st_datastream2.start_acquisition()
        st_device2.acquisition_start()
        remote_nodemap2 = st_device2.remote_port.nodemap
        set_enumeration(remote_nodemap2,"TriggerMode", "Off")
        error_cam2 = False
        return  st_datastream2, st_device2,remote_nodemap2
    except Exception as exception:     
        print('Error Cam 2:', exception)
        str_error = "Error"
        #sg.popup(str_error,font=('Helvetica',15), text_color='red',keep_on_top= True)
        window['result_cam2'].update(value= str_error, text_color='red')
'''
def time_to_name():
    current_time = datetime.datetime.now() 
    name_folder = str(current_time)
    name_folder = list(name_folder)
    for i in range(len(name_folder)):
        if name_folder[i] == ':':
            name_folder[i] = '-'
        if name_folder[i] == ' ':
            name_folder[i] ='_'
        if name_folder[i] == '.':
            name_folder[i] ='-'
    name_folder = ''.join(name_folder)
    return name_folder


def load_theme():
    name_themes = []
    with open('static/theme.txt') as lines:
        for line in lines:
            _, name_theme = line.strip().split(':')
            name_themes.append(name_theme)
    return name_themes

def load_choosemodel():
    with open('static/choose_model.txt') as lines:
        for line in lines:
            _, name_model = line.strip().split('=')
    return name_model

def save_theme(name_theme):
    line = 'theme:' + name_theme
    with open('static/theme.txt','w') as f:
        f.write(line)

def save_choosemodel(name_model):
    line = 'choose_model=' + name_model
    with open('static/choose_model.txt','w') as f:
        f.write(line)

def load_model(i):
    with open('static/model'+ str(i) + '.txt','r') as lines:
        for line in lines:
            _, name_model = line.strip().split('=')
    return name_model

def save_model(i,name_model):
    line = 'model' + str(i) + '=' + name_model
    with open('static/model' + str(i) + '.txt','w') as f:
        f.write(line)

def chk_conn(conn):
     try:
        conn.cursor()
        return True
     except Exception as ex:
        return False

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def load_all(model,i):
    values_all = []
    with open('static/all'+ str(i) + '.txt','r') as lines:
        for line in lines:
            _, name_all = line.strip().split('=')
            values_all.append(name_all)
    window['file_weights' + str(i)].update(value=values_all[0])
    window['conf_thres' + str(i)].update(value=values_all[1])
    a=1
    for item in range(len(model.names)):
        window[f'{model.names[item]}_' + str(i)].update(value=str2bool(values_all[a+1]))
        window[f'{model.names[item]}_OK_' + str(i)].update(value=str2bool(values_all[a+2]))
        window[f'{model.names[item]}_Num_' + str(i)].update(value=str(values_all[a+3]))
        window[f'{model.names[item]}_NG_' + str(i)].update(value=str2bool(values_all[a+4]))
        window[f'{model.names[item]}_Wn_' + str(i)].update(value=str(values_all[a+5]))
        window[f'{model.names[item]}_Wx_' + str(i)].update(value=str(values_all[a+6]))
        window[f'{model.names[item]}_Hn_' + str(i)].update(value=str(values_all[a+7]))
        window[f'{model.names[item]}_Hx_' + str(i)].update(value=str(values_all[a+8]))
        window[f'{model.names[item]}_PLC_' + str(i)].update(value=str(values_all[a+9]))
        window[f'{model.names[item]}_conf_' + str(i)].update(value=str(values_all[a+10]))
        a += 10


# def save_all(model,i):
#     with open('static/all'+ str(i) + '.txt','w') as f:
#         f.write('weights' + str(i) + '=' + str(values['file_weights' + str(i)]))
#         f.write('\n')
#         f.write('conf' + str(i) + '=' + str(values['conf_thres' + str(i)]))
#         f.write('\n')

#         for item in range(len(model.names)):
#             f.write(str(f'{model.names[item]}_' + str(i)) + '=' + str(values[f'{model.names[item]}_' + str(i)]))
#             f.write('\n')
#             f.write(str(f'{model.names[item]}_OK_' + str(i)) + '=' + str(values[f'{model.names[item]}_OK_' + str(i)]))
#             f.write('\n')
#             f.write(str(f'{model.names[item]}_Num_' + str(i)) + '=' + str(values[f'{model.names[item]}_Num_' + str(i)]))
#             f.write('\n')
#             f.write(str(f'{model.names[item]}_NG_' + str(i)) + '=' + str(values[f'{model.names[item]}_NG_' + str(i)]))
#             f.write('\n')
#             f.write(str(f'{model.names[item]}_Wn_' + str(i)) + '=' + str(values[f'{model.names[item]}_Wn_' + str(i)]))
#             f.write('\n')
#             f.write(str(f'{model.names[item]}_Wx_' + str(i)) + '=' + str(values[f'{model.names[item]}_Wx_' + str(i)]))
#             f.write('\n')
#             f.write(str(f'{model.names[item]}_Hn_' + str(i)) + '=' + str(values[f'{model.names[item]}_Hn_' + str(i)]))
#             f.write('\n')
#             f.write(str(f'{model.names[item]}_Hx_' + str(i)) + '=' + str(values[f'{model.names[item]}_Hx_' + str(i)]))
#             if item != len(model.names)-1:
#                 f.write('\n')


def load_all_sql(i,choose_model):
    conn = sqlite3.connect('2cam_3model.db')
    cursor = conn.execute("SELECT * from MYMODEL")
    for row in cursor:
        #if row[0] == values['choose_model']:
        if row[0] == choose_model:
            row1_a, row1_b = row[1].strip().split('_')
            if row1_a == str(i) and row1_b == '0':
                window['file_weights' + str(i)].update(value=row[2])
                window['conf_thres' + str(i)].update(value=row[3])
                window['have_save_OK_1'].update(value=str2bool(row[4]))
                window['have_save_OK_2'].update(value=str2bool(row[5]))
                window['have_save_OK_3'].update(value=str2bool(row[6]))
                window['have_save_NG_1'].update(value=str2bool(row[7]))
                window['have_save_NG_2'].update(value=str2bool(row[8]))
                window['have_save_NG_3'].update(value=str2bool(row[9]))

                window['save_OK_1'].update(value=row[10])
                window['save_OK_2'].update(value=row[11])
                window['save_OK_3'].update(value=row[12])
        
                window['save_NG_1'].update(value=row[13])
                window['save_NG_2'].update(value=row[14])
                window['save_NG_3'].update(value=row[15])
        
                model = torch.hub.load('./levu','custom', path= row[2], source='local',force_reload =False)
            if row1_a == str(i):                
                for item in range(len(model.names)):
                    if int(row1_b) == item:
                        window[f'{model.names[item]}_' + str(i)].update(value=str2bool(row[16]))
                        window[f'{model.names[item]}_OK_' + str(i)].update(value=str2bool(row[17]))
                        window[f'{model.names[item]}_Num_' + str(i)].update(value=str(row[18]))
                        window[f'{model.names[item]}_NG_' + str(i)].update(value=str2bool(row[19]))
                        window[f'{model.names[item]}_Wn_' + str(i)].update(value=str(row[20]))
                        window[f'{model.names[item]}_Wx_' + str(i)].update(value=str(row[21]))
                        window[f'{model.names[item]}_Hn_' + str(i)].update(value=str(row[22]))
                        window[f'{model.names[item]}_Hx_' + str(i)].update(value=str(row[23]))
                        window[f'{model.names[item]}_PLC_' + str(i)].update(value=str(row[24]))
                        window[f'OK_PLC_' + str(i)].update(value=str(row[25]))
                        window[f'{model.names[item]}_conf_' + str(i)].update(value=str(row[26]))
    
    conn.close()


def save_all_sql(model,i,choose_model):
    conn = sqlite3.connect('2cam_3model.db')
    cursor = conn.execute("SELECT * from MYMODEL")
    update = 0 
    answer = sg.popup_yes_no('Muon giu thong so cai dat')
    if answer == 'Yes':
        conn.execute('UPDATE MYMODEL SET Weights=?  WHERE (ChooseModel = ? AND Camera LIKE ?)',(str(values['file_weights' + str(i)]),choose_model,str(i) + '%'))          
        update = 1            
    if answer == 'No':
        for row in cursor:
            if row[0] == choose_model:            
                row1_a, _ = row[1].strip().split('_')
                if row1_a == str(i):
                    conn.execute("DELETE FROM MYMODEL WHERE (ChooseModel = ? AND Camera LIKE ?)", (choose_model,str(i) + '%'))
                    for item in range(len(model.names)):
                        conn.execute("INSERT INTO MYMODEL (ChooseModel,Camera, Weights,Confidence,OK_Cam1,OK_Cam2,OK_Cam3,NG_Cam1,NG_Cam2,NG_Cam3,Folder_OK_Cam1,Folder_OK_Cam2,Folder_OK_Cam3,Folder_NG_Cam1,Folder_NG_Cam2,Folder_NG_Cam3,Joined,Ok,Num,NG,WidthMin, WidthMax,HeightMin,HeightMax,PLC_NG,PLC_OK,Conf) \
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (str(values['choose_model']),str(i)+ '_' +str(item) ,str(values['file_weights' + str(i)]), int(values['conf_thres' + str(i)]),str(values['have_save_OK_1']),str(values['have_save_OK_2']),str(values['have_save_OK_3']),str(values['have_save_NG_1']),str(values['have_save_NG_2']),str(values['have_save_NG_3']),str(values['save_OK_1']),str(values['save_OK_2']),str(values['save_OK_3']),str(values['save_NG_1']),str(values['save_NG_2']),str(values['save_NG_3']),str(values[f'{model.names[item]}_' + str(i)]), str(values[f'{model.names[item]}_OK_' + str(i)]), int(values[f'{model.names[item]}_Num_' + str(i)]), str(values[f'{model.names[item]}_NG_' + str(i)]), int(values[f'{model.names[item]}_Wn_' + str(i)]), int(values[f'{model.names[item]}_Wx_' + str(i)]), int(values[f'{model.names[item]}_Hn_' + str(i)]), int(values[f'{model.names[item]}_Hx_' + str(i)]), int(values[f'{model.names[item]}_PLC_' + str(i)]), int(values['OK_PLC_' + str(i)]), int(values[f'{model.names[item]}_conf_' + str(i)])))           
                        update = 1
                    break

    if update == 0:
        for item in range(len(model.names)):
            conn.execute("INSERT INTO MYMODEL (ChooseModel,Camera, Weights,Confidence,OK_Cam1,OK_Cam2,OK_Cam3,NG_Cam1,NG_Cam2,NG_Cam3,Folder_OK_Cam1,Folder_OK_Cam2,Folder_OK_Cam3,Folder_NG_Cam1,Folder_NG_Cam2,Folder_NG_Cam3,Joined,Ok,Num,NG,WidthMin, WidthMax,HeightMin,HeightMax,PLC_NG,PLC_OK,Conf) \
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (str(values['choose_model']),str(i)+ '_' +str(item) ,str(values['file_weights' + str(i)]), int(values['conf_thres' + str(i)]),str(values['have_save_OK_1']),str(values['have_save_OK_2']),str(values['have_save_OK_3']),str(values['have_save_NG_1']),str(values['have_save_NG_2']),str(values['have_save_NG_3']),str(values['save_OK_1']),str(values['save_OK_2']),str(values['save_OK_3']),str(values['save_NG_1']),str(values['save_NG_2']),str(values['save_NG_3']),str(values[f'{model.names[item]}_' + str(i)]), str(values[f'{model.names[item]}_OK_' + str(i)]), int(values[f'{model.names[item]}_Num_' + str(i)]), str(values[f'{model.names[item]}_NG_' + str(i)]), int(values[f'{model.names[item]}_Wn_' + str(i)]), int(values[f'{model.names[item]}_Wx_' + str(i)]), int(values[f'{model.names[item]}_Hn_' + str(i)]), int(values[f'{model.names[item]}_Hx_' + str(i)]),int(values[f'{model.names[item]}_PLC_' + str(i)]), int(values['OK_PLC_' + str(i)]), int(values[f'{model.names[item]}_conf_' + str(i)])))
            
    for row in cursor:
        if row[0] == choose_model:
            conn.execute("UPDATE MYMODEL SET OK_Cam1 = ? , OK_Cam2 = ?,OK_Cam3 = ? , NG_Cam1 = ?,NG_Cam2 = ?, NG_Cam3 = ?, Folder_OK_Cam1 = ?, Folder_OK_Cam2 = ?,Folder_OK_Cam3 = ?, Folder_NG_Cam1 = ?, Folder_NG_Cam2 = ?,Folder_NG_Cam3 = ? WHERE ChooseModel = ? ",(str(values['have_save_OK_1']),str(values['have_save_OK_2']),str(values['have_save_OK_3']),str(values['have_save_NG_1']),str(values['have_save_NG_2']),str(values['have_save_NG_3']),str(values['save_OK_1']),str(values['save_OK_2']),str(values['save_OK_3']),str(values['save_NG_1']),str(values['save_NG_2']),str(values['save_NG_3']),choose_model))


    conn.commit()
    conn.close()
    load_all_sql(i,choose_model)


def program_camera1_FH(model,size,conf,regno):
    
    read_D = plc.read_word('D',regno)  # doc thanh ghi D450
    if read_D == 1:
        dir_path = 'C:/FH/CAM1/**/Input0_Camera0.jpg'
        filenames = glob(dir_path)
        if len(filenames) == 0:
            print('folder CAM1 empty')
        else:
            for filename1 in filenames:
                img1_orgin = cv2.imread(filename1)

                while type(img1_orgin) == type(None):
                    print('loading img 1...')
                    for filename1 in filenames:
                        img1_orgin = cv2.imread(filename1)


                #img1_orgin = cv2.imread(filename1)
                img1_save = img1_orgin
                t1 = time.time()

                # ghi vao D450 gia tri 0
                plc.write_word('D', regno, 0) 

                img1_orgin = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB)
                result1 = model(img1_orgin,size= size,conf = conf) 
                table1 = result1.pandas().xyxy[0]
                area_remove1 = []

                myresult1 =0                
                for item in range(len(table1.index)):
                    width1 = table1['xmax'][item] - table1['xmin'][item]
                    height1 = table1['ymax'][item] - table1['ymin'][item]
                    #area1 = width1*height1
                    label_name = table1['name'][item]
                    for i1 in range(len(model1.names)):
                        if values[f'{model1.names[i1]}_1'] == True:
                            #if values[f'{model1.names[i1]}_WH'] == True:
                            if label_name == model1.names[i1]:
                                if width1 < int(values[f'{model1.names[i1]}_Wn_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif width1 > int(values[f'{model1.names[i1]}_Wx_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif height1 < int(values[f'{model1.names[i1]}_Hn_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif height1 > int(values[f'{model1.names[i1]}_Hx_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                        if values[f'{model1.names[i1]}_1'] == False:
                            if label_name == model1.names[i1]:
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                names1 = list(table1['name'])

                show1 = np.squeeze(result1.render(area_remove1))
                show_1 = cv2.resize(show1, (720,540), interpolation = cv2.INTER_AREA)
                show_1 = cv2.cvtColor(show_1, cv2.COLOR_BGR2RGB)
                show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)
                k=1
                #ta = time.time()
                for i1 in range(len(model1.names)):
                    register_ng = int(values[f'{model1.names[i1]}_PLC_1'])
                    if values[f'{model1.names[i1]}_OK_1'] == True:
                        len_name1 = 0
                        for name1 in names1:
                            if name1 == model1.names[i1]:
                                len_name1 +=1
                        if len_name1 != int(values[f'{model1.names[i1]}_Num_1']):
                            print('NG 1(1)')
                            plc.write_word('D',register_ng,1)                            
                            t2 = time.time() - t1
                            print(t2) 
                            cv2.putText(show1,model1.names[i1],(5,30*k),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),0)
                            window['result_cam1'].update(value= 'NG', text_color='red')
                            if values['have_save_NG_1']:
                                name_folder_ng = time_to_name()
                                cv2.imwrite(values['save_NG_1']  + '/' + name_folder_ng + '.jpg',img1_save)
                            myresult1 = 1
                            k+=1                       
                    if values[f'{model1.names[i1]}_NG_1'] == True:
                        if model1.names[i1] in names1:
                            print('NG 2(1)')
                            plc.write_word('D',register_ng,1)
                            t2 = time.time() - t1
                            print(t2) 
                            cv2.putText(show1, model1.names[i1],(5,30*k),cv2.FONT_HERSHEY_COMPLEX, 1 ,(0,0,255),0)
                            window['result_cam1'].update(value= 'NG', text_color='red')    
                            if values['have_save_NG_1']:
                                name_folder_ng = time_to_name()
                                cv2.imwrite(values['save_NG_1']  + '/' + name_folder_ng + '.jpg',img1_save)
                            myresult1 = 1 
                            k+=1        
                
                if myresult1 == 0:
                    print('OK')                    
                    plc.write_word('D',int(values['OK_PLC_1']),1)
                    t2 = time.time() - t1
                    print(t2)                    
                    cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
                    window['result_cam1'].update(value= 'OK', text_color='green')
                    if values['have_save_OK_1']:
                        name_folder_ng = time_to_name()
                        cv2.imwrite(values['save_OK_1']  + '/' + name_folder_ng + '.jpg',img1_save)
                
                #Bao hoan tat CAM1
                plc.write_word('D',454,1)

                time_cam1 = str(int(t2*1000)) + 'ms'
                window['time_cam1'].update(value= time_cam1, text_color='black') 
            
                imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                window['image1'].update(data= imgbytes1)
                imgbytes1 = cv2.imencode('.png',show_1)[1].tobytes()
                window['toan1'].update(data= imgbytes1)   
                print(table1)  
                #Xoa thu muc
                #time.sleep(0.1)
                fname=os.path.dirname(filename1)
                shutil.rmtree(fname)                                                 
                
                print('CAM1')
                print('---------------------------------------------')

def program_camera2_FH(model,size,conf,file):
    
    img2_orgin = cv2.imread(file)

    img2_save = img2_orgin
    #edit
    t1 = time.time()
    
    img2_orgin = cv2.cvtColor(img2_orgin, cv2.COLOR_BGR2RGB)

    result2 = model(img2_orgin,size= size,conf = conf) 
    table2 = result2.pandas().xyxy[0]
    area_remove2 = []

    myresult2 =0 
    for item in range(len(table2.index)):
        width2 = table2['xmax'][item] - table2['xmin'][item]
        height2 = table2['ymax'][item] - table2['ymin'][item]
        label_name = table2['name'][item]
        for i2 in range(len(model2.names)):
            if values[f'{model2.names[i2]}_2'] == True:
                if label_name == model2.names[i2]:
                    if width2 < int(values[f'{model2.names[i2]}_Wn_2']): 
                        table2.drop(item, axis=0, inplace=True)
                        area_remove2.append(item)
                    elif width2 > int(values[f'{model2.names[i2]}_Wx_2']): 
                        table2.drop(item, axis=0, inplace=True)
                        area_remove2.append(item)
                    elif height2 < int(values[f'{model2.names[i2]}_Hn_2']): 
                        table2.drop(item, axis=0, inplace=True)
                        area_remove2.append(item)
                    elif height2 > int(values[f'{model2.names[i2]}_Hx_2']): 
                        table2.drop(item, axis=0, inplace=True)
                        area_remove2.append(item)

            if values[f'{model2.names[i2]}_2'] == False:
                if label_name == model2.names[i2]:
                    table2.drop(item, axis=0, inplace=True)
                    area_remove2.append(item)

    names2 = list(table2['name'])
    
    show2 = np.squeeze(result2.render(area_remove2))
    show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
    show2 = cv2.cvtColor(show2, cv2.COLOR_BGR2RGB)
    #ta = time.time()
    ng2 = True
    for i2 in range(len(model2.names)):
        error_number = int(values[f'{model2.names[i2]}_PLC_2'])

        if values[f'{model2.names[i2]}_NG_2'] == True:
            if model2.names[i2] in names2:
                print('NG 2')
                all_error.append(error_number)
                t2 = time.time() - t1
                print(t2) 
                cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                #window['result_cam2'].update(value= 'NG', text_color='red')    
                if values['have_save_NG_2']:
                    name_folder_ng = time_to_name()
                    cv2.imwrite(values['save_NG_2']  + '/' + name_folder_ng + '.jpg',img2_save)
                myresult2 = 1         
                ng2 = False

    for i2 in range(len(model2.names)):

        error_number = int(values[f'{model2.names[i2]}_PLC_2'])
        if values[f'{model2.names[i2]}_OK_2'] == True:
            len_name2 = 0
            for name2 in names2:
                if name2 == model2.names[i2]:
                    len_name2 +=1
            if len_name2 != int(values[f'{model2.names[i2]}_Num_2']):
                print('NG 1')
                
                if ng2:
                    all_error.append(error_number)
                t2 = time.time() - t1
                print(t2) 
                cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                #window['result_cam2'].update(value= 'NG', text_color='red')
                if values['have_save_NG_2']:
                    name_folder_ng = time_to_name()
                    cv2.imwrite(values['save_NG_2']  + '/' + name_folder_ng + '.jpg',img2_save)
                myresult2 = 1
                

    if myresult2 == 0:
        print('OK')        
        t2 = time.time() - t1
        print(t2) 
        cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
        #window['result_cam2'].update(value= 'OK', text_color='green')
        if values['have_save_OK_2']:
            name_folder_ng = time_to_name()
            cv2.imwrite(values['save_OK_2']  + '/' + name_folder_ng + '.jpg',img2_save)
    
    time_cam2 = str(int(t2*1000)) + 'ms'
    window['time_cam2'].update(value= time_cam2, text_color='black') 


    imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
    window['image2'].update(data= imgbytes2)
    
    #Xoa thu muc   
    fname=os.path.dirname(file)
    shutil.rmtree(fname)
    print('CAM2_CD')
    print('---------------------------------------------')
        


def program_camera3_FH(model,size,conf,file):
    
    img3_orgin = cv2.imread(file)

    img3_save = img3_orgin

    t1 = time.time()   
    
    img3_orgin = cv2.cvtColor(img3_orgin, cv2.COLOR_BGR2RGB)

    result3 = model(img3_orgin,size= size,conf = conf) 
    table3 = result3.pandas().xyxy[0]
    area_remove3 = []

    myresult3 =0        
    for item in range(len(table3.index)):
        width3 = table3['xmax'][item] - table3['xmin'][item]
        height3 = table3['ymax'][item] - table3['ymin'][item]
        label_name = table3['name'][item]
        for i3 in range(len(model3.names)):
            if values[f'{model3.names[i3]}_3'] == True:
                #if values[f'{model3.names[i3]}_WH'] == True:
                if label_name == model3.names[i3]:
                    if width3 < int(values[f'{model3.names[i3]}_Wn_3']): 
                        table3.drop(item, axis=0, inplace=True)
                        area_remove3.append(item)
                    elif width3 > int(values[f'{model3.names[i3]}_Wx_3']): 
                        table3.drop(item, axis=0, inplace=True)
                        area_remove3.append(item)
                    elif height3 < int(values[f'{model3.names[i3]}_Hn_3']): 
                        table3.drop(item, axis=0, inplace=True)
                        area_remove3.append(item)
                    elif height3 > int(values[f'{model3.names[i3]}_Hx_3']): 
                        table3.drop(item, axis=0, inplace=True)
                        area_remove3.append(item)

            if values[f'{model3.names[i3]}_3'] == False:
                if label_name == model3.names[i3]:
                    table3.drop(item, axis=0, inplace=True)
                    area_remove3.append(item)

    names3 = list(table3['name'])

    show3 = np.squeeze(result3.render(area_remove3))
    show3 = cv2.resize(show3, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
    show3 = cv2.cvtColor(show3, cv2.COLOR_BGR2RGB)
    #ta = time.time()
    global all_error
    ng2 = True
    for i3 in range(len(model3.names)):
        error_number = int(values[f'{model3.names[i3]}_PLC_3'])

        if values[f'{model3.names[i3]}_NG_3'] == True:
            if model3.names[i3] in names3:
                print('NG 2')
                all_error.append(error_number)
                t2 = time.time() - t1
                print(t2) 
                cv2.putText(show3, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                #window['result_cam2'].update(value= 'NG', text_color='red')
                if values['have_save_NG_3']:
                    name_folder_ng = time_to_name()
                    cv2.imwrite(values['save_NG_3']  + '/' + name_folder_ng + '.jpg',img3_save)
                myresult3 = 1     
                ng2 = False

    for i3 in range(len(model3.names)):
        error_number = int(values[f'{model3.names[i3]}_PLC_3'])

        if values[f'{model3.names[i3]}_OK_3'] == True:
            len_name3 = 0
            for name3 in names3:
                if name3 == model3.names[i3]:
                    len_name3 +=1
            if len_name3 != int(values[f'{model3.names[i3]}_Num_3']):
                print('NG 1')
                
                if ng2:
                    all_error.append(error_number)
                t2 = time.time() - t1
                print(t2) 
                cv2.putText(show3, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                #window['result_cam2'].update(value= 'NG', text_color='red')
                if values['have_save_NG_3']:
                    name_folder_ng = time_to_name()
                    cv2.imwrite(values['save_NG_3']  + '/' + name_folder_ng + '.jpg',img3_save)
                myresult3 = 1
                    
                

    if myresult3 == 0:
        print('OK')        
        t2 = time.time() - t1
        print(t2) 
        cv2.putText(show3, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
        #window['result_cam2'].update(value= 'OK', text_color='green')
        if values['have_save_OK_3']:
            name_folder_ng = time_to_name()
            cv2.imwrite(values['save_OK_3']  + '/' + name_folder_ng + '.jpg',img3_save)
        
    time_cam3 = str(int(t2*1000)) + 'ms'
    window['time_cam2'].update(value= time_cam3, text_color='black') 


    imgbytes3 = cv2.imencode('.png',show3)[1].tobytes()
    window['image2'].update(data= imgbytes3)


    #Xoa thu muc    
    fname=os.path.dirname(file)
    shutil.rmtree(fname)
    print('CAM2_TC')
    print('---------------------------------------------')
     
def hien_thi(pics):
    gimg = os.path.basename(pics)
    t1 = time.time()
    mod_no = values['opt_model']
    if mod_no == 'opt1':
        result1 = model1(pics, size=int(values['co_immg']), conf = values['conf_thres1']/100)
    if mod_no == 'opt2':
        result1 = model2(pics, size=int(values['co_immg']), conf = values['conf_thres2']/100)
    if mod_no == 'opt3':
        result1 = model3(pics, size=int(values['co_immg']), conf = values['conf_thres3']/100)    
    table1 = result1.pandas().xyxy[0]
    area_remove1 = []
    
    if not chk_conn(conn2):
        sg.popup('Chua ket noi duoc CSDL')

    myresult1 =0 
    txt =''
    for item in range(len(table1.index)):
        width1 = table1['xmax'][item] - table1['xmin'][item]
        height1 = table1['ymax'][item] - table1['ymin'][item]
        #area1 = width1*height1
        conf1 = table1['confidence'][item]
        label_name = table1['name'][item]
       
        txt = 'hekko ' + str(int(conf1*100)) + " " + str(int(width1)) + ' ' + str(int(height1)) + ' '
        if mod_no == 'opt1':
            for i1 in range(len(model1.names)):
                if values[f'{model1.names[i1]}_1'] == True:
                    #if values[f'{model1.names[i1]}_WH'] == True:
                    if label_name == model1.names[i1]:
                        if conf1 < int(values[f'{model1.names[i1]}_conf_1'])/100: 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        if width1 < int(values[f'{model1.names[i1]}_Wn_1']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif width1 > int(values[f'{model1.names[i1]}_Wx_1']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif height1 < int(values[f'{model1.names[i1]}_Hn_1']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif height1 > int(values[f'{model1.names[i1]}_Hx_1']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                if values[f'{model1.names[i1]}_1'] == False:
                    if label_name == model1.names[i1]:
                        table1.drop(item, axis=0, inplace=True)
                        area_remove1.append(item)
        if mod_no == 'opt2':
            for i2 in range(len(model2.names)):
                if values[f'{model2.names[i2]}_2'] == True:
                    #if values[f'{model1.names[i1]}_WH'] == True:
                    if label_name == model2.names[i2]:
                        if conf1 < int(values[f'{model2.names[i2]}_conf_2'])/100: 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        if width1 < int(values[f'{model2.names[i2]}_Wn_2']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif width1 > int(values[f'{model2.names[i2]}_Wx_2']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif height1 < int(values[f'{model2.names[i2]}_Hn_2']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif height1 > int(values[f'{model2.names[i2]}_Hx_2']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                if values[f'{model2.names[i2]}_2'] == False:
                    if label_name == model2.names[i2]:
                        table1.drop(item, axis=0, inplace=True)
                        area_remove1.append(item)
        if mod_no == 'opt3':
            for i3 in range(len(model3.names)):
                if values[f'{model3.names[i3]}_3'] == True:
                    #if values[f'{model3.names[i3]}_WH'] == True:
                    if label_name == model3.names[i3]:
                        if conf1 < int(values[f'{model3.names[i3]}_conf_3'])/100: 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif width1 < int(values[f'{model3.names[i3]}_Wn_3']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif width1 > int(values[f'{model3.names[i3]}_Wx_3']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif height1 < int(values[f'{model3.names[i3]}_Hn_3']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                        elif height1 > int(values[f'{model3.names[i3]}_Hx_3']): 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove1.append(item)
                if values[f'{model3.names[i3]}_3'] == False:
                    if label_name == model3.names[i3]:
                        table1.drop(item, axis=0, inplace=True)
                        area_remove1.append(item)

    names1 = list(table1['name'])

    show1 = np.squeeze(result1.render(area_remove1))
    show_1 = cv2.resize(show1, (int(720*1.3),int(540*1.3)), interpolation = cv2.INTER_AREA)
    show_1 = cv2.cvtColor(show_1, cv2.COLOR_BGR2RGB)
    #ta = time.time()
    hm=[]
    if mod_no == 'opt1':
        for i1 in range(len(model1.names)):
            if values[f'{model1.names[i1]}_1'] == True:
                if values[f'{model1.names[i1]}_OK_1'] == True:
                    len_name1 = 0
                    for name1 in names1:
                        if name1 == model1.names[i1]:
                            len_name1 +=1
                    if len_name1 != int(values[f'{model1.names[i1]}_Num_1']):
                        hm.append(model1.names[i1])                                                                           
                        myresult1 = 1
                        #break

                if values[f'{model1.names[i1]}_NG_1'] == True:
                    if model1.names[i1] in names1:
                        hm.append(model1.names[i1])
                        myresult1 = 1         
                        #break 
    if mod_no == 'opt2':
        for i2 in range(len(model2.names)):
            if values[f'{model2.names[i2]}_2'] == True:
                if values[f'{model2.names[i2]}_OK_2'] == True:
                    len_name1 = 0
                    for name1 in names1:
                        if name1 == model2.names[i2]:
                            len_name1 +=1
                    if len_name1 != int(values[f'{model2.names[i2]}_Num_2']):
                        hm.append(model2.names[i2])                                                                           
                        myresult1 = 1
                        #break

                if values[f'{model2.names[i2]}_NG_2'] == True:
                    if model2.names[i2] in names1:
                        hm.append(model2.names[i2])
                        myresult1 = 1         
                        #break 
    if mod_no == 'opt3':
        for i3 in range(len(model3.names)):
            if values[f'{model3.names[i3]}_3'] == True:
                if values[f'{model3.names[i3]}_OK_3'] == True:
                    len_name1 = 0
                    for name1 in names1:
                        if name1 == model3.names[i3]:
                            len_name1 +=1
                    if len_name1 != int(values[f'{model3.names[i3]}_Num_3']):
                        hm.append(model3.names[i3])                                                                           
                        myresult1 = 1
                        #break

                if values[f'{model3.names[i3]}_NG_3'] == True:
                    if model3.names[i3] in names1:
                        hm.append(model3.names[i3])
                        myresult1 = 1         
                        #break 
    fimg = time_to_name()
    k=1
    if myresult1 == 0:
        print('OK')
        cv2.putText(show_1, 'OK',(600,100),cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),3)
    else:
        for i in hm:
            print('NG')
            cv2.putText(show_1,i,(30,50*k),cv2.FONT_HERSHEY_COMPLEX,  1,(0,0,255),0)
            k+=1
        #print(*hm, sep = ", ") 
        toan = ', '.join(hm)
        #print(fimg)
        print('NG',toan)
        try:
            conn2.execute('INSERT INTO PHEPHAM (ANH, HANGMUC, CHITIET) VALUES (?,?,?)', (gimg[:-4], toan, txt))
            conn2.commit()
        except:
            #sg.popup('Da test', gimg)
            pass
        
    t2 = time.time() - t1
    window['tgxl'].update(value=str(int(t2*1000))+'ms')
    imgbytes1 = cv2.imencode('.png',show_1)[1].tobytes()
    window['toan1'].update(data= imgbytes1)
    print(table1)
    # print(area_remove1)
    #window.TKroot.title(pics.replace('/','\\') + ' [' + str(dem) + '/' + str(ttl) + ']')
    window['DDAN'].update(value=pics.replace('/','\\') + ' [' + str(dem) + '/' + str(ttl) + ']')
    print('--------------------')

def make_window(theme):
    sg.theme(theme)

    #file_img = [("JPEG (*.jpg)",("*jpg","*.png"))]

    file_weights = [('Weights (*.pt)', ('*.pt'))]

    # menu = [['Application', ['Connect PLC','Interrupt Connect PLC','Exit']],
    #         ['Tool', ['Check Cam','Change Theme']],
    #         ['Help',['About']]]

    right_click_menu = [[], ['Exit','Administrator','Change Theme']]

    layout_main = [

        [
        sg.Text('CAM 1',justification='center' ,font= ('Helvetica',30),text_color='red', expand_y=True),
        sg.Text('CAM 2',justification='center' ,font= ('Helvetica',30),text_color='red',expand_x=True),
        ],
        # sg.Frame('',[
        #     [sg.Text('CAM 2',justification='center' ,font= ('Helvetica',30),text_color='red'),
        #     sg.Text('CAM 1',justification='center' ,font= ('Helvetica',30),text_color='red')],
        # ]),

        [
        #1
        sg.Frame('',[
            #[sg.Image(filename='', size=(640,480),key='image1',background_color='black')],
            [sg.Image(filename='', size=(image_width_display,image_height_display),key='image1',background_color='black')],
            [sg.Frame('',
            [
                [sg.Text('',font=('Helvetica',90), justification='center', key='result_cam1',expand_x=True)],
                [sg.Text('',font=('Helvetica',30), justification='center', key='time_cam1', expand_x=True)],
            ], vertical_alignment='top',size=(int(560*0.5),int(250*0.5))),
            sg.Frame('',[
                #[sg.Text('')],
                #[sg.Button('Webcam', size=(8,1),  font=('Helvetica',14),disabled=True ,key= 'Webcam1')],
                #[sg.Text('')],
                [sg.Button('Stop', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'Stop1')],
                [sg.Text('')],
                #[sg.Button('Continue', size=(8,1),  font=('Helvetica',14), disabled=True, key= 'Continue1')],
                #[sg.Text('')],
                #[sg.Button('Snap', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'Snap1')],
                #[sg.Text('')],
                [sg.Checkbox('Check1',size=(6,1),font=('Helvetica',14), key='check_model1',enable_events=True,expand_x=True, expand_y=True)],
                #],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),
                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),
                
            sg.Frame('',[   
                #[sg.Button('Change', size=(8,1), font=('Helvetica',14), disabled= True, key= 'Change1')],
                #[sg.Text('')],
                [sg.Button('Pic', size=(8,1), font=('Helvetica',14),disabled=True,key= 'Pic1')],
                #[sg.Text('')],
                [sg.Button('Detect', size=(8,1), font=('Helvetica',14),disabled=True,key= 'Detect1')],
                #[sg.Text('',size=(4,2))],
                #[sg.Text('',size=(4,1))],
                [sg.Combo(values=['1','3','4','5','6','7','8','9'], default_value='3',font=('Helvetica',20),size=(5, 100),text_color='navy',enable_events= True, key='choose_model'),],
                #[sg.Button('SaveData', size=(8,1), font=('Helvetica',14),disabled=True,key= 'SaveData1')],
                [sg.Text('',size=(4,1))],
                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),
            ],
                
        ]),
    
        # 2
        sg.Frame('',[
            #[sg.Image(filename='', size=(640,480),key='image1',background_color='black')],
            [sg.Image(filename='', size=(image_width_display,image_height_display),key='image2',background_color='black')],
            [sg.Frame('',
            [
                [sg.Text('',font=('Helvetica',90), justification='center', key='result_cam2',expand_x=True)],
                [sg.Text('',font=('Helvetica',30),justification='center', key='time_cam2',expand_x=True)],
            ], vertical_alignment='top',size=(int(560*0.5),int(250*0.5))),
            sg.Frame('',[
                #[sg.Text('')],
                #[sg.Button('Webcam', size=(8,1),  font=('Helvetica',14),disabled=True,key= 'Webcam2',auto_size_button=True)],
                #[sg.Text('')],
                [sg.Button('Stop', size=(8,1), font=('Helvetica',14),disabled=True  ,key='Stop2')],
                [sg.Text('')],
                #[sg.Button('Continue', size=(8,1),  font=('Helvetica',14),disabled=True ,key='Continue2')],
                #[sg.Text('')],
                #[sg.Button('Snap', size=(8,1), font=('Helvetica',14),disabled=True  ,key='Snap2')],
                #[sg.Text('')],
                [sg.Checkbox('Taychoi',size=(6,1),font=('Helvetica',14), key='Tay_choi',enable_events=True,expand_x=True,expand_y=True)],
                [sg.Checkbox('Check2',size=(6,1),font=('Helvetica',14), key='check_model2',enable_events=True,expand_y=True)],
                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),

            sg.Frame('',[   
                #[sg.Button('Change', size=(8,1), font=('Helvetica',14), disabled= True, key= 'Change2')],
                #[sg.Text('')],
                [sg.Button('Pic', size=(8,1), font=('Helvetica',14),disabled=True,key='Pic2')],
                #[sg.Text('')],
                [sg.Button('Detect', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'Detect2')],
                [sg.Text('',size=(4,2))],
                [sg.Text('',size=(4,1))],
                #[sg.Button('SaveData', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'SaveData2')],

                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),
            ]
        ], expand_y= True),

    ]] 

    layout_cam1 = [
       [
        sg.Frame('PATH',[
            [sg.Image(filename = '', key='toan1'),
            sg.Checkbox('AutoCheck',size=(10,6),default=False,font=('Helvetica',15),  key='autochk',enable_events=True, disabled=False), 
            sg.Text('Delay',size=(5,1),font=('Helvetica',15)),
            sg.Input('1',size=(2,1),font=('Helvetica',15),key= 'lagtime',text_color='navy',enable_events=True, disabled=False),
            sg.Text(' ',size=(3,1)),
            sg.Text('0 ms',size=(10,1),font=('Helvetica',35),key='tgxl')],
            [
            sg.Frame('', [   
                [sg.Text('Folder'), sg.In(size=(60,1), enable_events=True , disabled= True, key='folder_browse0'), sg.FolderBrowse()],]), 
            sg.Combo(('opt1','opt2','opt3'),default_value='opt1', font=('Helvetica',12), key='opt_model'),
            sg.Button('PREV', size=(12,1), font=('Helvetica',10), disabled= False, key= 'run0'),
            sg.Input('1', size=(4,1), font=('Helvetica',12), enable_events=True, key= 'sott', disabled= True), 
            sg.Button('NEXT', size=(12,1), font=('Helvetica',10), disabled= False, key= 'run1'),
            sg.Combo(('416','768','608'),default_value='416', font=('Helvetica',12), key='co_immg'),
            sg.Button('OutCSV', size=(10,1), font=('Helvetica',10), disabled= False, key= 'xuatCSV')],
            ], relief= sg.RELIEF_FLAT, font=('Helvetica',20), key='DDAN'),
    ]]

    layout_cam2 = [
        
        [
        #1
        sg.Frame('CHAU DIEN',[
            [sg.Image(filename='', size=(600,450),key='cd1',background_color='black')],
            [sg.Image(filename='', size=(600,450),key='cd2',background_color='black')],    
        ]),
    
        # 2
        sg.Frame('TAY CHOI',[
            [sg.Image(filename='', size=(600,450),key='tc1',background_color='black')],
            [sg.Image(filename='', size=(600,450),key='tc2',background_color='black')],
        ], expand_y= True),

    ]] 
   

    layout_option1_0 = [
        [sg.Frame('',[
        [sg.Frame('',
        [   
            [sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color='red'), sg.Input(size=(80,1), font=('Helvetica',12), key='file_weights1',readonly= True, text_color='navy',enable_events= True),
            sg.Frame('',[
                [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse1',enable_events=True, disabled=True)]
            ], relief= sg.RELIEF_FLAT),
            sg.Frame('',[
                [sg.Button('Change Model', size=(14,1), font=('Helvetica',10), disabled= True, key= 'Change_1')]
            ], relief= sg.RELIEF_FLAT),],
            [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='red'), sg.Slider(range=(1,100),orientation='h',size=(80,20),font=('Helvetica',11),disabled=True, key= 'conf_thres1'),]
        ], relief=sg.RELIEF_FLAT),
        ],
        [sg.Frame('',[
            [
            sg.Text('Name',size=(15,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Join',size=(5,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('OK',size=(4,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Num',size=(4,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('NG',size=(4,1),font=('Helvetica',15), text_color='red'),  
            sg.Text('W_Min',size=(7,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('W_Max',size=(12,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('H_Min',size=(8,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('H_Max',size=(9,1),font=('Helvetica',15), text_color='red'),
            sg.Text('PLC',size=(8,1),font=('Helvetica',15), text_color='red'),
            sg.Text(' Confidence detail',size=(25,1), font=('Helvetica',15), text_color='red')],
        ], relief=sg.RELIEF_FLAT)],
        ])]
    ]
    
    layout_option1_1 = [
        [sg.Frame('',[
        
        [sg.Frame('',[
            [
                sg.Text(f'{model1.names[i1]}_1',size=(16,1),font=('Helvetica',15), text_color='yellow'), 
                sg.Checkbox('',size=(2,5),default=True,font=('Helvetica',15),  key=f'{model1.names[i1]}_1',enable_events=True, disabled=True), 
                # sg.Radio('',group_id=f'Cam1 {i1}',size=(1,5),default=False,font=('Helvetica',15),  key=f'{model1.names[i1]}_OK_1',enable_events=True, disabled=True), 
                sg.Checkbox('',size=(1,5),default=False,font=('Helvetica',15),  key=f'{model1.names[i1]}_OK_1',enable_events=True, disabled=True), 
                
                sg.Input('1',size=(3,1),justification='center',font=('Helvetica',15),key= f'{model1.names[i1]}_Num_1',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(1,1),font=('Helvetica',15), text_color='red'), 
                # sg.Radio('',group_id=f'Cam1 {i1}',size=(1,5),default=False,font=('Helvetica',15),  key=f'{model1.names[i1]}_NG_1',enable_events=True, disabled=True), 
                sg.Checkbox('',size=(1,5),default=False,font=('Helvetica',15),  key=f'{model1.names[i1]}_NG_1',enable_events=True, disabled=True), 
                
                sg.Input('0',size=(4,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Wn_1',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('1600',size=(6,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Wx_1',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(5,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('0',size=(4,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Hn_1',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('1200',size=(6,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Hx_1',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('30',size=(4,1),font=('Helvetica',15),key= f'{model1.names[i1]}_PLC_1',text_color='navy',enable_events=True, disabled=True),
                sg.Slider(range=(1,100), orientation='h',size=(25,10),font=('Helvetica',11),enable_events=True,disabled=True, key= f'{model1.names[i1]}_conf_1'), 
            ] for i1 in range(len(model1.names))
        ], relief=sg.RELIEF_FLAT)],
 
        ])]
    ]


    layout_option1_2 = [
        [sg.Frame('',[

        [sg.Text('  OK',size=(16,1),font=('Helvetica',15), text_color='yellow'),
        sg.Text('_ '*62), 
        sg.Input('0',size=(4,1),font=('Helvetica',15),key= 'OK_PLC_1',text_color='navy',enable_events=True)],
        [sg.Text(' ')],
        [sg.Text(' '*250), sg.Button('Save Data', size=(12,1),  font=('Helvetica',12),key='SaveData1',enable_events=True),
        sg.Text('             '),] 
        ])]
    ]
    
    layout_option2_0 = [
        [sg.Frame('',[
        [sg.Frame('',
        [   
            [sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color='red'), sg.Input(size=(80,1), font=('Helvetica',12), key='file_weights2',readonly= True, text_color='navy',enable_events= True),
            sg.Frame('',[
                [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse2',enable_events=True, disabled=True)]
            ], relief= sg.RELIEF_FLAT),
            sg.Frame('',[
                [sg.Button('Change Model', size=(14,1), font=('Helvetica',10), disabled= True, key= 'Change_2')]
            ], relief= sg.RELIEF_FLAT),],
            [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='red'), sg.Slider(range=(1,100),orientation='h',size=(80,20),font=('Helvetica',11),disabled=True, key= 'conf_thres2'),]
        ], relief=sg.RELIEF_FLAT),
        ],
        [sg.Frame('',[
            [
            sg.Text('Name',size=(15,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Join',size=(5,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('OK',size=(4,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Num',size=(4,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('NG',size=(4,1),font=('Helvetica',15), text_color='red'),  
            sg.Text('W_Min',size=(7,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('W_Max',size=(12,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('H_Min',size=(8,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('H_Max',size=(9,1),font=('Helvetica',15), text_color='red'),
            sg.Text('PLC',size=(8,1),font=('Helvetica',15), text_color='red'),
            sg.Text(' Confidence detail',size=(25,1), font=('Helvetica',15), text_color='red')],
        ], relief=sg.RELIEF_FLAT)],
        ])]
    ]
    
    layout_option2_1 = [
        [sg.Frame('',[
        
        [sg.Frame('',[
            [
                sg.Text(f'{model2.names[i2]}_2',size=(16,1),font=('Helvetica',15), text_color='yellow'), 
                sg.Checkbox('',size=(2,5),default=True,font=('Helvetica',15),  key=f'{model2.names[i2]}_2',enable_events=True, disabled=True), 
                sg.Radio('',group_id=f'Cam1 {i2}',size=(1,5),default=False,font=('Helvetica',15),  key=f'{model2.names[i2]}_OK_2',enable_events=True, disabled=True), 
                sg.Input('1',size=(3,1),justification='center',font=('Helvetica',15),key= f'{model2.names[i2]}_Num_2',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(1,1),font=('Helvetica',15), text_color='red'), 
                sg.Radio('',group_id=f'Cam1 {i2}',size=(1,5),default=False,font=('Helvetica',15),  key=f'{model2.names[i2]}_NG_2',enable_events=True, disabled=True), 
                sg.Input('0',size=(4,1),font=('Helvetica',15),key= f'{model2.names[i2]}_Wn_2',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('1600',size=(6,1),font=('Helvetica',15),key= f'{model2.names[i2]}_Wx_2',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(5,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('0',size=(4,1),font=('Helvetica',15),key= f'{model2.names[i2]}_Hn_2',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('1200',size=(6,1),font=('Helvetica',15),key= f'{model2.names[i2]}_Hx_2',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('30',size=(4,1),font=('Helvetica',15),key= f'{model2.names[i2]}_PLC_2',text_color='navy',enable_events=True, disabled=True),
                sg.Slider(range=(1,100), orientation='h',size=(25,10),font=('Helvetica',11),enable_events=True,disabled=True, key= f'{model2.names[i2]}_conf_2'), 
            ] for i2 in range(len(model2.names))
        ], relief=sg.RELIEF_FLAT)],
 
        ])]
    ]


    layout_option2_2 = [
        [sg.Frame('',[

        [sg.Text('  OK',size=(16,1),font=('Helvetica',15), text_color='yellow'),
        sg.Text('_ '*62), 
        sg.Input('0',size=(4,1),font=('Helvetica',15),key= 'OK_PLC_2',text_color='navy',enable_events=True)],
        [sg.Text(' ')],
        [sg.Text(' '*250), sg.Button('Save Data', size=(12,1),  font=('Helvetica',12),key='SaveData2',enable_events=True),
        sg.Text('             '),] 
        ])]
    ]


    layout_option3_0 = [
        [sg.Frame('',[
        [sg.Frame('',
        [   
            [sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color='red'), sg.Input(size=(80,1), font=('Helvetica',12), key='file_weights3',readonly= True, text_color='navy',enable_events= True),
            sg.Frame('',[
                [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse3',enable_events=True, disabled=True)]
            ], relief= sg.RELIEF_FLAT),
            sg.Frame('',[
                [sg.Button('Change Model', size=(14,1), font=('Helvetica',10), disabled= True, key= 'Change_3')]
            ], relief= sg.RELIEF_FLAT),],
            [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='red'), sg.Slider(range=(1,100),orientation='h',size=(80,20),font=('Helvetica',11),disabled=True, key= 'conf_thres3'),]
        ], relief=sg.RELIEF_FLAT),
        ],
        [sg.Frame('',[
            [
            sg.Text('Name',size=(15,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Join',size=(5,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('OK',size=(4,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Num',size=(4,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('NG',size=(4,1),font=('Helvetica',15), text_color='red'),  
            sg.Text('W_Min',size=(7,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('W_Max',size=(12,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('H_Min',size=(8,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('H_Max',size=(9,1),font=('Helvetica',15), text_color='red'),
            sg.Text('PLC',size=(8,1),font=('Helvetica',15), text_color='red'),
            sg.Text(' Confidence detail',size=(25,1), font=('Helvetica',15), text_color='red')],
        ], relief=sg.RELIEF_FLAT)],
        ])]
    ]
    
    layout_option3_1 = [
        [sg.Frame('',[
        
        [sg.Frame('',[
            [
                sg.Text(f'{model3.names[i3]}_3',size=(16,1),font=('Helvetica',15), text_color='yellow'), 
                sg.Checkbox('',size=(2,5),default=True,font=('Helvetica',15),  key=f'{model3.names[i3]}_3',enable_events=True, disabled=True), 
                sg.Radio('',group_id=f'Cam1 {i3}',size=(1,5),default=False,font=('Helvetica',15),  key=f'{model3.names[i3]}_OK_3',enable_events=True, disabled=True), 
                sg.Input('1',size=(3,1),justification='center',font=('Helvetica',15),key= f'{model3.names[i3]}_Num_3',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(1,1),font=('Helvetica',15), text_color='red'), 
                sg.Radio('',group_id=f'Cam1 {i3}',size=(1,5),default=False,font=('Helvetica',15),  key=f'{model3.names[i3]}_NG_3',enable_events=True, disabled=True), 
                sg.Input('0',size=(4,1),font=('Helvetica',15),key= f'{model3.names[i3]}_Wn_3',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('1600',size=(6,1),font=('Helvetica',15),key= f'{model3.names[i3]}_Wx_3',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(5,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('0',size=(4,1),font=('Helvetica',15),key= f'{model3.names[i3]}_Hn_3',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('1200',size=(6,1),font=('Helvetica',15),key= f'{model3.names[i3]}_Hx_3',text_color='navy',enable_events=True, disabled=True), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('30',size=(4,1),font=('Helvetica',15),key= f'{model3.names[i3]}_PLC_3',text_color='navy',enable_events=True, disabled=True),
                sg.Slider(range=(1,100), orientation='h',size=(25,10),font=('Helvetica',11),enable_events=True,disabled=True, key= f'{model3.names[i3]}_conf_3'), 
            ] for i3 in range(len(model3.names))
        ], relief=sg.RELIEF_FLAT)],
 
        ])]
    ]


    layout_option3_2 = [
        [sg.Frame('',[

        [sg.Text('  OK',size=(16,1),font=('Helvetica',15), text_color='yellow'),
        sg.Text('_ '*62), 
        sg.Input('0',size=(4,1),font=('Helvetica',15),key= 'OK_PLC_3',text_color='navy',enable_events=True)],
        [sg.Text(' ')],
        [sg.Text(' '*250), sg.Button('Save Data', size=(12,1),  font=('Helvetica',12),key='SaveData3',enable_events=True),
        sg.Text('             '),] 
        ])]
    ]

    layout_savimg = [
        [sg.Frame('',[
                [sg.Text('Have save folder image OK for camera 1',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=False,font=('Helvetica',15),  key='have_save_OK_1',enable_events=True, disabled=True)], 
                [sg.T('Choose folder save image OK for camera 1', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/Cam1/OK' ,font=('Helvetica',12), key='save_OK_1',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_OK_1',enable_events=True) ],
                [sg.Text('')],
                [sg.Text('Have save folder image OK for camera 2',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=False,font=('Helvetica',15),  key='have_save_OK_2',enable_events=True, disabled=True)], 
                [sg.T('Choose folder save image OK for camera 2', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/Cam2/OK' , font=('Helvetica',12), key='save_OK_2',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_OK_2',enable_events=True) ],
                [sg.Text('')],
                [sg.Text('Have save folder image NG for camera 1',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_NG_1',enable_events=True, disabled=True)], 
                [sg.T('Choose folder save image NG for camera 1', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/Cam1/NG' , font=('Helvetica',12), key='save_NG_1',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_NG_1',enable_events=True) ],
                [sg.Text('')],
                [sg.Text('Have save folder image NG for camera 2',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_NG_2',enable_events=True, disabled=True)], 
                [sg.T('Choose folder save image NG for camera 2', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/Cam2/NG' , font=('Helvetica',12), key='save_NG_2',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_NG_2',enable_events=True) ],
        ], relief=sg.RELIEF_FLAT),
        sg.Frame('',[
                [sg.Text('Have save folder image OK for camera 3',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=False,font=('Helvetica',15),  key='have_save_OK_3',enable_events=True, disabled=True)], 
                [sg.T('Choose folder save image OK for camera 3', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/Cam3/OK' ,font=('Helvetica',12), key='save_OK_3',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_OK_3',enable_events=True) ],
                [sg.Text('')],
                [sg.Text('Have save folder image NG for camera 3',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_NG_3',enable_events=True, disabled=True)], 
                [sg.T('Choose folder save image NG for camera 3', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/Cam3/NG' , font=('Helvetica',12), key='save_NG_3',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_NG_3',enable_events=True) ],
        ], relief=sg.RELIEF_FLAT)],
        ]
    layout_terminal = [[sg.Text("Anything printed will display here!")],
                      [sg.Multiline( font=('Helvetica',14), write_only=True, autoscroll=True, auto_refresh=True,reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True,expand_x=True,expand_y=True)]
                      ]
    layout_option1 = [[sg.Column(layout_option1_0)],
                        [sg.Column(layout_option1_1, expand_y= True, scrollable = True, vertical_scroll_only=True)],
                        [sg.Column(layout_option1_2,expand_y=True)],
    
    ]
    layout_option2 = [[sg.Column(layout_option2_0)],
                        [sg.Column(layout_option2_1, size=(1200,350), scrollable = True, vertical_scroll_only=True)],
                        [sg.Column(layout_option2_2,expand_y=True)],
    
    ]
    layout_option3 = [[sg.Column(layout_option3_0)],
                        [sg.Column(layout_option3_1, size=(1200,350), scrollable = True, vertical_scroll_only=True)],
                        [sg.Column(layout_option3_2,expand_y=True)],
    
    ]
    layout = [[sg.TabGroup([[  sg.Tab('Main', layout_main),
                               sg.Tab('Cam1', layout_cam1),
                               sg.Tab('Cam2', layout_cam2),
                               sg.Tab('Option for model 1',[[sg.Column(layout_option1, expand_y= True)],]),
                               sg.Tab('Option for model 2',[[sg.Column(layout_option2, expand_y= True)],]),
                               sg.Tab('Option for model 3',[[sg.Column(layout_option3, expand_y= True)],]),
                               sg.Tab('Save Image', layout_savimg),
                               sg.Tab('Output', layout_terminal)]])
               ]]

    #layout[-1].append(sg.Sizegrip())
    window = sg.Window('HuynhLeVu', layout, location=(0,0),right_click_menu=right_click_menu,resizable=True).Finalize()
    #window.bind('<Configure>',"Configure")
    window.Maximize()

    return window


image_width_display = int(760*0.8)
image_height_display = int(480*0.8)

result_width_display = 680
result_height_display = 100 


file_name_img = [("Img(*.jpg,*.png)",("*jpg","*.png"))]


recording1 = False
recording2 = False 

error_cam1 = True
error_cam2 = True

recording3 = False

error_cam3 = True

#window['result_cam1'].update(value= 'Wait', text_color='yellow')
#window['result_cam2'].update(value= 'Wait', text_color='yellow')


connected = True
while connected == False:
    print('connecting....')
    connected = plc.socket_connect('192.168.250.20', 8000)
print("connected plc")  

mypath1 = load_model(1)
model1 = torch.hub.load('./levu','custom', path= mypath1, source='local',force_reload =False)

img1_test = os.path.join(os.getcwd(), 'img/imgtest.jpg')
result1 = model1(img1_test,416,0.25) 
print('model1 already')

mypath2 = load_model(2)
model2 = torch.hub.load('./levu','custom', path= mypath2, source='local',force_reload =False)

img2_test = os.path.join(os.getcwd(), 'img/imgtest.jpg')
result2 = model2(img2_test,416,0.25) 

print('model2 already')

mypath3 = load_model(3)
model3 = torch.hub.load('./levu','custom', path= mypath3, source='local',force_reload =False)

img3_test = os.path.join(os.getcwd(), 'img/imgtest.jpg')
result3 = model3(img1_test,416,0.25) 
print('model3 already')

choose_model = load_choosemodel()

themes = load_theme()
theme = themes[0]
window = make_window(theme)

window['choose_model'].update(value=choose_model)
conn2 = sqlite3.connect('2cam_3model.db')  

try:
    load_all_sql(1,choose_model)
except:
    print(traceback.format_exc())
    window['time_cam1'].update(value= "Error data") 


try:
    load_all_sql(2,choose_model)
except:
    print(traceback.format_exc())
    window['time_cam2'].update(value= "Error data") 

try:
    load_all_sql(3,choose_model)
except:
    print(traceback.format_exc())
    window['time_cam2'].update(value= "Error data") 



connect_camera1 = False
connect_camera2 = False
connect_camera3 = False

connect_total = False


if connect_camera1 == True and connect_total == True:
    window['result_cam1'].update(value= 'Done', text_color='blue')
if connect_camera2 == True and connect_total == True:
    window['result_cam2'].update(value= 'Done', text_color='blue')

#Reset 
#plc.write_word('D',450,0)
#plc.write_word('D',460,0)

removefile()
#Bao cho PLC reset all
#plc.write_word('D',490,1)
xem=[]
dem = 0
ttl = 0
chk=False
try:
    while True:
        event, values = window.read(timeout=20)
        
        for i1 in range(len(model1.names)):
            #if event == f'{model1.names[i1]}_1':
            if values[f'{model1.names[i1]}_1'] == False:
                window[f'{model1.names[i1]}_OK_1'].update(disabled=True)
                window[f'{model1.names[i1]}_Num_1'].update(disabled=True)
                window[f'{model1.names[i1]}_NG_1'].update(disabled=True)
                window[f'{model1.names[i1]}_Wn_1'].update(disabled=True)
                window[f'{model1.names[i1]}_Wx_1'].update(disabled=True)
                window[f'{model1.names[i1]}_Hn_1'].update(disabled=True)
                window[f'{model1.names[i1]}_Hx_1'].update(disabled=True)
                window[f'{model1.names[i1]}_PLC_1'].update(disabled=True)
                window[f'{model1.names[i1]}_conf_1'].update(disabled=True)

            elif values[f'{model1.names[i1]}_1'] == True:
                window[f'{model1.names[i1]}_OK_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Num_1'].update(disabled=False)
                window[f'{model1.names[i1]}_NG_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Wn_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Wx_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Hn_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Hx_1'].update(disabled=False)
                window[f'{model1.names[i1]}_PLC_1'].update(disabled=False)
                window[f'{model1.names[i1]}_conf_1'].update(disabled=False)

        for i1 in range(len(model1.names)):
            if event == f'{model1.names[i1]}_OK_1':
                if values[f'{model1.names[i1]}_OK_1'] == True:
                    window[f'{model1.names[i1]}_NG_1'].update(disabled=True)
                else:
                    window[f'{model1.names[i1]}_NG_1'].update(disabled=False)
            if event == f'{model1.names[i1]}_NG_1':
                if values[f'{model1.names[i1]}_NG_1'] == True:
                    window[f'{model1.names[i1]}_OK_1'].update(disabled=True)
                else:
                    window[f'{model1.names[i1]}_OK_1'].update(disabled=False)


        for i2 in range(len(model2.names)):
            #if event == f'{model2.names[i2]}_2':
            if values[f'{model2.names[i2]}_2'] == False:
                window[f'{model2.names[i2]}_OK_2'].update(disabled=True)
                window[f'{model2.names[i2]}_Num_2'].update(disabled=True)
                window[f'{model2.names[i2]}_NG_2'].update(disabled=True)
                window[f'{model2.names[i2]}_Wn_2'].update(disabled=True)
                window[f'{model2.names[i2]}_Wx_2'].update(disabled=True)
                window[f'{model2.names[i2]}_Hn_2'].update(disabled=True)
                window[f'{model2.names[i2]}_Hx_2'].update(disabled=True)
                window[f'{model2.names[i2]}_PLC_2'].update(disabled=True)
                window[f'{model2.names[i2]}_conf_2'].update(disabled=True)

            elif values[f'{model2.names[i2]}_2'] == True:
                window[f'{model2.names[i2]}_OK_2'].update(disabled=False)
                window[f'{model2.names[i2]}_Num_2'].update(disabled=False)
                window[f'{model2.names[i2]}_NG_2'].update(disabled=False)
                window[f'{model2.names[i2]}_Wn_2'].update(disabled=False)
                window[f'{model2.names[i2]}_Wx_2'].update(disabled=False)
                window[f'{model2.names[i2]}_Hn_2'].update(disabled=False)
                window[f'{model2.names[i2]}_Hx_2'].update(disabled=False)
                window[f'{model2.names[i2]}_PLC_2'].update(disabled=False)
                window[f'{model2.names[i2]}_conf_2'].update(disabled=False)

        for i2 in range(len(model2.names)):
            if event == f'{model2.names[i2]}_OK_2':
                if values[f'{model2.names[i2]}_OK_2'] == True:
                    window[f'{model2.names[i2]}_NG_2'].update(disabled=True)
                else:
                    window[f'{model2.names[i2]}_NG_2'].update(disabled=False)
            if event == f'{model2.names[i2]}_NG_2':
                if values[f'{model2.names[i2]}_NG_2'] == True:
                    window[f'{model2.names[i2]}_OK_2'].update(disabled=True)
                else:
                    window[f'{model2.names[i2]}_OK_2'].update(disabled=False)

        for i3 in range(len(model3.names)):
            #if event == f'{model3.names[i3]}_3':
            if values[f'{model3.names[i3]}_3'] == False:
                window[f'{model3.names[i3]}_OK_3'].update(disabled=True)
                window[f'{model3.names[i3]}_Num_3'].update(disabled=True)
                window[f'{model3.names[i3]}_NG_3'].update(disabled=True)
                window[f'{model3.names[i3]}_Wn_3'].update(disabled=True)
                window[f'{model3.names[i3]}_Wx_3'].update(disabled=True)
                window[f'{model3.names[i3]}_Hn_3'].update(disabled=True)
                window[f'{model3.names[i3]}_Hx_3'].update(disabled=True)
                window[f'{model3.names[i3]}_PLC_3'].update(disabled=True)
                window[f'{model3.names[i3]}_conf_3'].update(disabled=True)

            elif values[f'{model3.names[i3]}_3'] == True:
                window[f'{model3.names[i3]}_OK_3'].update(disabled=False)
                window[f'{model3.names[i3]}_Num_3'].update(disabled=False)
                window[f'{model3.names[i3]}_NG_3'].update(disabled=False)
                window[f'{model3.names[i3]}_Wn_3'].update(disabled=False)
                window[f'{model3.names[i3]}_Wx_3'].update(disabled=False)
                window[f'{model3.names[i3]}_Hn_3'].update(disabled=False)
                window[f'{model3.names[i3]}_Hx_3'].update(disabled=False)
                window[f'{model3.names[i3]}_PLC_3'].update(disabled=False)
                window[f'{model3.names[i3]}_conf_3'].update(disabled=False)

        for i3 in range(len(model3.names)):
            if event == f'{model3.names[i3]}_OK_3':
                if values[f'{model3.names[i3]}_OK_3'] == True:
                    window[f'{model3.names[i3]}_NG_3'].update(disabled=True)
                else:
                    window[f'{model3.names[i3]}_NG_3'].update(disabled=False)
            if event == f'{model3.names[i3]}_NG_3':
                if values[f'{model3.names[i3]}_NG_3'] == True:
                    window[f'{model3.names[i3]}_OK_3'].update(disabled=True)
                else:
                    window[f'{model3.names[i3]}_OK_3'].update(disabled=False)



        if event =='Exit' or event == sg.WINDOW_CLOSED :
            break

        # if event == 'Configure':
        #     if window.TKroot.state() == 'zoomed':
        #         #print(window['image1'].get_size()[0])
        #         image_width_display = window['image1'].get_size()[0]
        #         image_height_display = window['image1'].get_size()[1]
        #         result_width_display = image_width_display - 190
        #         result_height_display = 100 


        if event =='Administrator':
            login_password = 'q'  # helloworld
            password = sg.popup_get_text(
                'Enter PassworE: ', password_char='*') 
            if password == login_password:
                sg.popup_ok('Login Successed!!! ',text_color='green', font=('Helvetica',14))  

                window['conf_thres2'].update(disabled= False)
                window['conf_thres1'].update(disabled= False)

                window['file_browse2'].update(disabled= False,button_color='turquoise')
                window['file_browse1'].update(disabled= False,button_color='turquoise')

                window['SaveData1'].update(disabled= False,button_color='turquoise')
                window['SaveData2'].update(disabled= False,button_color='turquoise')

                #window['Webcam1'].update(disabled= True,button_color='turquoise')
                #window['Webcam2'].update(disabled= True,button_color='turquoise')
                window['Stop1'].update(disabled= False,button_color='turquoise')
                window['Stop2'].update(disabled= False,button_color='turquoise')
                window['Pic1'].update(disabled= False,button_color='turquoise')
                window['Pic2'].update(disabled= False,button_color='turquoise')
                #window['Snap1'].update(disabled= True,button_color='turquoise')
                #window['Snap2'].update(disabled= True,button_color='turquoise')
                #window['Change1'].update(button_color='turquoise')
                #window['Change2'].update(button_color='turquoise')
                window['Change_1'].update(button_color='turquoise')
                window['Change_2'].update(button_color='turquoise')
                window['Detect1'].update(button_color='turquoise')
                window['Detect2'].update(button_color='turquoise')


                window['have_save_OK_1'].update(disabled=False)
                window['have_save_NG_1'].update(disabled=False)
                window['have_save_OK_2'].update(disabled=False)
                window['have_save_NG_2'].update(disabled=False)

                window['save_OK_1'].update(disabled=False)
                window['save_NG_1'].update(disabled=False)
                window['save_OK_2'].update(disabled=False)
                window['save_NG_2'].update(disabled=False)

                window['save_folder_OK_1'].update(disabled= False,button_color='turquoise')
                window['save_folder_NG_1'].update(disabled= False,button_color='turquoise')
                window['save_folder_OK_2'].update(disabled= False,button_color='turquoise')
                window['save_folder_NG_2'].update(disabled= False,button_color='turquoise')


                for i1 in range(len(model1.names)):
                    window[f'{model1.names[i1]}_1'].update(disabled=False)
                    window[f'{model1.names[i1]}_OK_1'].update(disabled=False)
                    window[f'{model1.names[i1]}_Num_1'].update(disabled=False)
                    window[f'{model1.names[i1]}_NG_1'].update(disabled=False)
                    window[f'{model1.names[i1]}_Wn_1'].update(disabled=False)
                    window[f'{model1.names[i1]}_Wx_1'].update(disabled=False)
                    window[f'{model1.names[i1]}_Hn_1'].update(disabled=False)
                    window[f'{model1.names[i1]}_Hx_1'].update(disabled=False)
                    window[f'{model1.names[i1]}_PLC_1'].update(disabled=False)
                    window[f'{model1.names[i1]}_conf_1'].update(disabled=False)

                for i2 in range(len(model2.names)):
                    window[f'{model2.names[i2]}_2'].update(disabled=False)
                    window[f'{model2.names[i2]}_OK_2'].update(disabled=False)
                    window[f'{model2.names[i2]}_Num_2'].update(disabled=False)
                    window[f'{model2.names[i2]}_NG_2'].update(disabled=False)
                    window[f'{model2.names[i2]}_Wn_2'].update(disabled=False)
                    window[f'{model2.names[i2]}_Wx_2'].update(disabled=False)
                    window[f'{model2.names[i2]}_Hn_2'].update(disabled=False)
                    window[f'{model2.names[i2]}_Hx_2'].update(disabled=False)
                    window[f'{model2.names[i2]}_PLC_2'].update(disabled=False)



                window['conf_thres3'].update(disabled= False)

                window['file_browse3'].update(disabled= False,button_color='turquoise')

                window['SaveData3'].update(disabled= False,button_color='turquoise')



                window['have_save_OK_3'].update(disabled=False)
                window['have_save_NG_3'].update(disabled=False)
                

                window['save_OK_3'].update(disabled=False)
                window['save_NG_3'].update(disabled=False)


                window['save_folder_OK_3'].update(disabled= False,button_color='turquoise')
                window['save_folder_NG_3'].update(disabled= False,button_color='turquoise')


                for i3 in range(len(model3.names)):
                    window[f'{model3.names[i3]}_3'].update(disabled=False)
                    window[f'{model3.names[i3]}_OK_3'].update(disabled=False)
                    window[f'{model3.names[i3]}_Num_3'].update(disabled=False)
                    window[f'{model3.names[i3]}_NG_3'].update(disabled=False)
                    window[f'{model3.names[i3]}_Wn_3'].update(disabled=False)
                    window[f'{model3.names[i3]}_Wx_3'].update(disabled=False)
                    window[f'{model3.names[i3]}_Hn_3'].update(disabled=False)
                    window[f'{model3.names[i3]}_Hx_3'].update(disabled=False)
                    window[f'{model3.names[i3]}_PLC_3'].update(disabled=False)


            else:
                sg.popup_cancel('Wrong Password!!!',text_color='red', font=('Helvetica',14))


        if event == 'Change Theme':
            layout_theme = [
                [sg.Listbox(values= sg.theme_list(), size = (30,20),auto_size_text=18,default_values='Dark',key='theme', enable_events=True)],
                [
                    [sg.Button('Apply'),
                    sg.Button('Cancel')]
                ]
            ] 
            window_theme = sg.Window('Change Theme', layout_theme, location=(50,50),keep_on_top=True).Finalize()
            window_theme.set_min_size((300,400))   

            while True:
                event_theme, values_theme = window_theme.read(timeout=20)
                if event_theme == sg.WIN_CLOSEE:
                    break

                if event_theme == 'Apply':
                    theme_choose = values_theme['theme'][0]
                    if theme_choose == 'Default':
                        continue
                    window.close()
                    window = make_window(theme_choose)
                    save_theme(theme_choose)
                    #print(theme_choose)
                if event_theme == 'Cancel':
                    answer = sg.popup_yes_no('Do you want to exit?')
                    if answer == 'Yes':
                        break
                    if answer == 'No':
                        continue
            window_theme.close()

        if event == 'folder_browse0':
            xem = glob(values['folder_browse0'] + '/*.jpg')
            #Xoa PHEPHAM
            ttl = len(xem)
            #conn = sqlite3.connect('2cam_3model.db')
            conn2.execute('DELETE FROM PHEPHAM')
            conn2.commit()
            dem = 1
            hien_thi(xem[dem-1])
            window['sott'].update(value = dem, disabled= False)

        if event == 'sott':
            try:
                dem = int(values['sott'])
            except:
                dem = 1
                window['sott'].update(value= dem)
            if dem < 1 or dem > ttl:
                sg.popup('Out of range')
                dem = 1
                window['sott'].update(value= dem)
            hien_thi(xem[dem-1])

        # if event == 'run0'  or keyboard.is_pressed('a'):
        if event == 'run0'  :
            if dem > 1:
                dem = int(values['sott']) - 1
                window['sott'].update(value= dem)
                hien_thi(xem[dem-1])
            else:
                sg.popup('Hinh dau tien')

        
        # if event == 'run0'  or keyboard.is_pressed('a'):
        if event == 'run1' :
            if dem < ttl:
                dem = int(values['sott']) + 1
                window['sott'].update(value= dem)
                hien_thi(xem[dem-1])
            else:
                sg.popup('Hinh cuoi cung')
        
        if event == 'xuatCSV':
            conn = sqlite3.connect('2cam_3model.db')
            df = pd.read_sql_query("select * from PHEPHAM", conn)
            fcsv = values['folder_browse0'] +'/' + time_to_name() + '.csv'
            df.to_csv(fcsv)
            conn.close()
            sg.popup('Duong dan:\n' + fcsv, title="Thong bao da xuat CSV")

        if event == 'autochk':
            chk = values['autochk']
            # for i in xem:
            #     if check_image(i) == "loi":
            #         continue
            #     else : 
            while dem < ttl and chk:
                hien_thi(xem[dem])
                dem += 1
                time.sleep(float(values['lagtime']))
                sk, gt = window.read(timeout=20)
                if keyboard.is_pressed('z'):
                    gt['autochk']==False
                if keyboard.is_pressed('x'):
                    gt['autochk']==True
                if gt['autochk']==False:
                    chk=False
                if gt['autochk']==True:
                    chk=True
                window['sott'].update(value= dem)
                
                        

        if keyboard.is_pressed('c'):
            image_width_display +=40
            image_height_display +=20

        if keyboard.is_pressed('t'):
            image_width_display -=40
            image_height_display -=20
            
        if event == 'runall':    
            tm = str(values['file_weights0']) + '/*.jpg'
            for pic0 in glob(tm):
                result1  = model1(pic0,size= 416,conf = values['conf_thres1']/100)

                table1 = result1.pandas().xyxy[0]
                print(table1)
                area_remove1 = []

                myresult1 =0 

                for item in range(len(table1.index)):
                    width1 = table1['xmax'][item] - table1['xmin'][item]
                    height1 = table1['ymax'][item] - table1['ymin'][item]
                    #area1 = width1*height1
                    label_name = table1['name'][item]
                    for i1 in range(len(model1.names)):
                        if values[f'{model1.names[i1]}_1'] == True:
                            #if values[f'{model1.names[i1]}_WH'] == True:
                            if label_name == model1.names[i1]:
                                if width1 < int(values[f'{model1.names[i1]}_Wn_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif width1 > int(values[f'{model1.names[i1]}_Wx_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif height1 < int(values[f'{model1.names[i1]}_Hn_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif height1 > int(values[f'{model1.names[i1]}_Hx_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                        if values[f'{model1.names[i1]}_1'] == False:
                            if label_name == model1.names[i1]:
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                names1 = list(table1['name'])

                show1 = np.squeeze(result1.render(area_remove1))
                show_1 = cv2.resize(show1, (720,540), interpolation = cv2.INTER_AREA)
                show_1 = cv2.cvtColor(show_1, cv2.COLOR_BGR2RGB)
                #ta = time.time()
                for i1 in range(len(model1.names)):
                    if values[f'{model1.names[i1]}_1'] == True:
                        if values[f'{model1.names[i1]}_OK_1'] == True:
                            len_name1 = 0
                            for name1 in names1:
                                if name1 == model1.names[i1]:
                                    len_name1 +=1
                            if len_name1 != int(values[f'{model1.names[i1]}_Num_1']):
                                print('NG')                                                                           
                                myresult1 = 1
                                break

                        if values[f'{model1.names[i1]}_NG_1'] == True:
                            if model1.names[i1] in names1:
                                print('NG')
                                myresult1 = 1         
                                break    
                k = 1
                if myresult1 == 0:
                    print('OK')
                    cv2.putText(show_1, 'OK',(1000,100),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
                else:
                    cv2.putText(show_1, 'NG',(1000,100),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
                    k +=1 
                    break 
                                                        
                imgbytes1 = cv2.imencode('.png',show_1)[1].tobytes()
                window['toan1'].update(data= imgbytes1)
                
                time.sleep(2)
                                        

        if event == 'file_browse1': 
            window['file_weights1'].update(value=values['file_browse1'])
            if values['file_browse1']:
                #window['Change1'].update(disabled=False)
                window['Change_1'].update(disabled=False)
                'Code them back_up database'
                conn = sqlite3.connect('2cam_3model.db')
                conn.execute("UPDATE MYMODEL Set Weights ='" + values['file_browse1'] + "' WHERE ChooseModel='1' And Camera Like '1%'")
                # conn.execute("DROP TABLE if exists BKMODEL")
                # conn.execute("CREATE TABLE BKMODEL AS SELECT * FROM MYMODEL")
                # conn.execute("DROP TABLE if exists TEMP")
                # conn.execute("CREATE TABLE TEMP as SELECT A.ChooseModel, A.Camera, '" + str(values['file_browse1']) + "' as Weights, A.Confidence, A.OK_Cam1, A.OK_Cam2, A.OK_Cam3, A.NG_Cam1, A.NG_Cam2, A.NG_Cam3, A.Folder_OK_Cam1, A.Folder_OK_Cam2, A.Folder_OK_Cam3, A.Folder_NG_Cam1, A.Folder_NG_Cam2, A.Folder_NG_Cam3, A.Joined, A.Ok, A.Num, A.NG, A.WidthMin, A. WidthMax, A.HeightMin, A.HeightMax, A.PLC_NG, A.PLC_OK FROM BKMODEL as A INNER JOIN MYMODEL as B ON A.ChooseModel = B.ChooseModel And A.Camera = B.Camera WHERE B.ChooseModel='1' And B.Camera Like '1%'")
                # conn.execute("DELETE FROM MYMODEL WHERE ChooseModel='1' And Camera Like '1%'")
                # conn.execute("REPLACE INTO MYMODEL SELECT * FROM TEMP")
                conn.commit()
                conn.close()



        if event == 'file_browse2':
            window['file_weights2'].update(value=values['file_browse2'])
            if values['file_browse2']:
                #window['Change2'].update(disabled=False)
                window['Change_2'].update(disabled=False)


        if event == 'file_browse3': 
            window['file_weights3'].update(value=values['file_browse3'])
            if values['file_browse3']:
                #window['Change3'].update(disabled=False)
                window['Change_3'].update(disabled=False)
        
        # change_chooose_model = plc.read_word('D', 400)
        # print(change_chooose_model)
        # if change_chooose_model == 0:
        #     print('den',values['choose_model'])
        #     values['choose_model'] = '3'
        # if change_chooose_model == 2:
        #     print('trang',values['choose_model'])
        #     values['choose_model'] = '1'
        if event == 'choose_model':
            mychoose = values['choose_model']
            weight1 = ''
            conf_thres1 = 1
            weight2 = ''
            conf_thres2 = 1

            OK_Cam1 = False
            OK_Cam2 = False
            NG_Cam1 = True
            NG_Cam2 = True
            Folder_OK_Cam1 = 'C:/Cam1/OK'
            Folder_OK_Cam2 = 'C:/Cam2/OK'
            Folder_NG_Cam1 = 'C:/Cam1/NG'
            Folder_NG_Cam2 = 'C:/Cam2/NG'

            weight3 = ''
            conf_thres3 = 1

            OK_Cam3 = False

            NG_Cam3 = True

            Folder_OK_Cam3 = 'C:/Cam3/OK'
            Folder_ = 'C://OK'
            Folder_NG_Cam3 = 'C:/Cam3/NG'
            Folder_NG_ = 'C://NG'
            conn = sqlite3.connect('2cam_3model.db')
            cursor = conn.execute("SELECT * from MYMODEL")
            for row in cursor:
                if row[0] == values['choose_model']:
 
                    mychoose = values['choose_model']
                    row1_a, row1_b = row[1].strip().split('_')
                    if row1_a == '1' and row1_b == '0':
                        weight1 = row[2]
                        conf_thres1 = row[3]
                        OK_Cam1 = str2bool(row[4])
                        OK_Cam2 = str2bool(row[5])
                        OK_Cam3 = str2bool(row[6])
                        NG_Cam1 = str2bool(row[7])
                        NG_Cam2 = str2bool(row[8])
                        NG_Cam3 = str2bool(row[9])
                        Folder_OK_Cam1 = row[10]
                        Folder_OK_Cam2 = row[11]
                        Folder_OK_Cam3 = row[12]
                        Folder_NG_Cam1 = row[13]
                        Folder_NG_Cam2 = row[14]
                        Folder_NG_Cam3 = row[15]
                        model1 = torch.hub.load('./levu','custom', path= row[2], source='local',force_reload =False)

                    if row1_a == '2' and row1_b == '0':
                        weight2 = row[2]
                        conf_thres2 = row[3]
                        OK_Cam1 = str2bool(row[4])
                        OK_Cam2 = str2bool(row[5])
                        OK_Cam3 = str2bool(row[6])

                        NG_Cam1 = str2bool(row[7])
                        NG_Cam2 = str2bool(row[8])
                        NG_Cam3 = str2bool(row[9])
            
                        Folder_OK_Cam1 = row[10]
                        Folder_OK_Cam2 = row[11]
                        Folder_OK_Cam3 = row[12]
             
                        Folder_NG_Cam1 = row[13]
                        Folder_NG_Cam2 = row[14]
                        Folder_NG_Cam3 = row[15]
            
                        model2 = torch.hub.load('./levu','custom', path= row[2], source='local',force_reload =False)

                    if row1_a == '3' and row1_b == '0':
                        weight3 = row[2]
                        conf_thres3 = row[3]
                        OK_Cam1 = str2bool(row[4])
                        OK_Cam2 = str2bool(row[5])
                        OK_Cam3 = str2bool(row[6])
               
                        NG_Cam1 = str2bool(row[7])
                        NG_Cam2 = str2bool(row[8])
                        NG_Cam3 = str2bool(row[9])
                      
                        Folder_OK_Cam1 = row[10]
                        Folder_OK_Cam2 = row[11]
                        Folder_OK_Cam3 = row[12]
           
                        Folder_NG_Cam1 = row[13]
                        Folder_NG_Cam2 = row[14]
                        Folder_NG_Cam3 = row[15]
                   
                        model3 = torch.hub.load('./levu','custom', path= row[2], source='local',force_reload =False)

                    if row1_a == '4' and row1_b == '0':
                        weight4 = row[2]
                        #window['conf_thres2'].update(value=row[3])
                        conf_thres4 = row[3]
                        OK_Cam1 = str2bool(row[4])
                        OK_Cam2 = str2bool(row[5])
                        OK_Cam3 = str2bool(row[6])
                        
                        NG_Cam1 = str2bool(row[7])
                        NG_Cam2 = str2bool(row[8])
                        NG_Cam3 = str2bool(row[9])
                  
                        Folder_OK_Cam1 = row[10]
                        Folder_OK_Cam2 = row[11]
                        Folder_OK_Cam3 = row[12]
                    
                        Folder_NG_Cam1 = row[13]
                        Folder_NG_Cam2 = row[14]
                        Folder_NG_Cam3 = row[15]
               
                        model4 = torch.hub.load('./levu','custom', path= row[2], source='local',force_reload =False)
        
            window.close() 
            window = make_window(theme)
            print(weight1)
            window['file_weights1'].update(value=weight1)
            window['conf_thres1'].update(value=conf_thres1)
            window['file_weights2'].update(value=weight2)
            window['conf_thres2'].update(value=conf_thres2)
            window['choose_model'].update(value=mychoose)

            window['have_save_OK_1'].update(value=OK_Cam1)
            window['have_save_OK_2'].update(value=OK_Cam2)
            window['have_save_NG_1'].update(value=NG_Cam1)
            window['have_save_NG_2'].update(value=NG_Cam2)

            window['save_OK_1'].update(value=Folder_OK_Cam1)
            window['save_OK_2'].update(value=Folder_OK_Cam2)
            window['save_NG_1'].update(value=Folder_NG_Cam1)
            window['save_NG_2'].update(value=Folder_NG_Cam2)

            window['file_weights3'].update(value=weight3)
            window['conf_thres3'].update(value=conf_thres3)

            window['choose_model'].update(value=mychoose)

            window['have_save_OK_3'].update(value=OK_Cam3)
  
            window['have_save_NG_3'].update(value=NG_Cam3)


            window['save_OK_3'].update(value=Folder_OK_Cam3)

            window['save_NG_3'].update(value=Folder_NG_Cam3)
   


            cursor = conn.execute("SELECT ChooseModel,Camera,Weights,Confidence,OK_Cam1,OK_Cam2,OK_Cam3,NG_Cam1,NG_Cam2,NG_Cam3,Folder_OK_Cam1,Folder_OK_Cam2,Folder_OK_Cam3,Folder_NG_Cam1,Folder_NG_Cam2,Folder_NG_Cam3,Joined,Ok,Num,NG,WidthMin,WidthMax,HeightMin,HeightMax,PLC_NG,PLC_OK from MYMODEL")
            for row in cursor:
                if row[0] == values['choose_model']:
                    row1_a, row1_b = row[1].strip().split('_')
                    if row1_a == '1':
                        for item in range(len(model1.names)):
                            if int(row1_b) == item:
                                window[f'{model1.names[item]}_1'].update(value=str2bool(row[16]))
                                window[f'{model1.names[item]}_OK_1'].update(value=str2bool(row[17]))
                                window[f'{model1.names[item]}_Num_1'].update(value=str(row[18]))
                                window[f'{model1.names[item]}_NG_1'].update(value=str2bool(row[19]))
                                window[f'{model1.names[item]}_Wn_1'].update(value=str(row[20]))
                                window[f'{model1.names[item]}_Wx_1'].update(value=str(row[21]))
                                window[f'{model1.names[item]}_Hn_1'].update(value=str(row[22]))
                                window[f'{model1.names[item]}_Hx_1'].update(value=str(row[23]))
                                window[f'{model1.names[item]}_PLC_1'].update(value=str(row[24]))
                                window['OK_PLC_1'].update(value=str(row[25]))

                    if row1_a == '2':
                        for item in range(len(model2.names)):
                            if int(row1_b) == item:
                                window[f'{model2.names[item]}_2'].update(value=str2bool(row[16]))
                                window[f'{model2.names[item]}_OK_2'].update(value=str2bool(row[17]))
                                window[f'{model2.names[item]}_Num_2'].update(value=str(row[18]))
                                window[f'{model2.names[item]}_NG_2'].update(value=str2bool(row[19]))
                                window[f'{model2.names[item]}_Wn_2'].update(value=str(row[20]))
                                window[f'{model2.names[item]}_Wx_2'].update(value=str(row[21]))
                                window[f'{model2.names[item]}_Hn_2'].update(value=str(row[22]))
                                window[f'{model2.names[item]}_Hx_2'].update(value=str(row[23]))
                                window[f'{model2.names[item]}_PLC_2'].update(value=str(row[24]))
                                window['OK_PLC_2'].update(value=str(row[25]))
                    if row1_a == '3':
                        for item in range(len(model3.names)):
                            if int(row1_b) == item:
                                window[f'{model3.names[item]}_3'].update(value=str2bool(row[16]))
                                window[f'{model3.names[item]}_OK_3'].update(value=str2bool(row[17]))
                                window[f'{model3.names[item]}_Num_3'].update(value=str(row[18]))
                                window[f'{model3.names[item]}_NG_3'].update(value=str2bool(row[19]))
                                window[f'{model3.names[item]}_Wn_3'].update(value=str(row[20]))
                                window[f'{model3.names[item]}_Wx_3'].update(value=str(row[21]))
                                window[f'{model3.names[item]}_Hn_3'].update(value=str(row[22]))
                                window[f'{model3.names[item]}_Hx_3'].update(value=str(row[23]))
                                window[f'{model3.names[item]}_PLC_3'].update(value=str(row[24]))
                                window['OK_PLC_3'].update(value=str(row[25]))
            conn.close()

        if event == 'SaveData1':
            save_all_sql(model1,1,str(values['choose_model']))
            save_choosemodel(str(values['choose_model']))
            save_model(1,values['file_weights1'])
            sg.popup('Saved param model 1 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


        if event == 'SaveData2':
            save_all_sql(model2,2,str(values['choose_model']))
            save_choosemodel(str(values['choose_model']))
            save_model(2,values['file_weights2'])
            sg.popup('Saved param model 2 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


        if event == 'SaveData3':
            save_all_sql(model3,3,str(values['choose_model']))
            save_choosemodel(str(values['choose_model']))
            save_model(3,values['file_weights3'])
            sg.popup('Saved param model 3 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)
        '''
        #Xuly CAM1
        program_camera1_FH(model=model1,size=416,conf= values['conf_thres1']/100, regno=450)
        
        #Xuly CAM2        
        read_D = plc.read_word('D',460) 
        if read_D == 1:   
            window['result_cam2'].update(value= ' ', text_color='green')
            
            all_error = []
            folder = glob('C:/FH/CAM2/**/Input0_Camera0.jpg')
            print(len(folder))
            if len(folder) >= 4:
                item = 0
                for f in folder:
                    #Chau dien 1
                    if item==0:
                        program_camera2_FH(model=model2,size=416,conf=values['conf_thres2']/100, file = f)     
                    #Tay choi 1
                    if item ==1:      
                        program_camera3_FH(model=model3,size=416,conf=values['conf_thres2']/100, file = f)
                    #Chau dien 2
                    if item==2:
                        program_camera2_FH(model=model2,size=416,conf=values['conf_thres2']/100, file = f)     
                    #Tay choi 2
                    if item ==3:      
                        program_camera3_FH(model=model3,size=416,conf=values['conf_thres2']/100, file = f)
                    if item >3:  
                        print('Delete file', item + 1)
                        fname1=os.path.dirname(f)
                        shutil.rmtree(fname1)                     
                    item +=1
                   
            
            all_error = set(all_error)
            all_error = list(all_error)
            print(all_error)
            if len(all_error) == 0:
                plc.write_word('D',430,int(values['OK_PLC_2']))
                window['result_cam2'].update(value= 'OK', text_color='green')
            if len(all_error) == 1:
                plc.write_word('D',430,int(all_error[0]))
                window['result_cam2'].update(value= 'NG', text_color='red')
            if len(all_error) >=2:
                plc.write_word('D',430,11) # nhieu hang muc
                window['result_cam2'].update(value= 'NG*', text_color='red')
                print(all_error)
                for error in all_error:
                    plc.write_word('D',460 + error*2 ,1)
                
            
            plc.write_word('D',460,0)

            plc.write_word('D',456,1) # hoan tat
        '''
              

                #Chau dien 2        
                #program_camera2_FH(model=model2,size=416,conf=values['conf_thres2']/100, regno=464)
                #Tay choi 2        
                #program_camera3_FH(model=model3,size=416,conf=values['conf_thres2']/100, regno=466)
                                            
                               
        #Shutdown if completed
        # read_500 = plc.read_word('D',500)
        # if read_500==10:
        #     os.system('shutdown -s -t 60')
        #     break

        #task_camera1(model=model1,size= 416,conf= values['conf_thres1']/100)
        #task_camera2(model=model2,size= 416,conf= values['conf_thres2']/100)

        #test_camera1(model=model1,size= 416,conf= values['conf_thres1']/100)
        #test_camera2()

        #task1(model1,size= 416,conf= values['conf_thres1']/100)
        #task2(model2,size= 416,conf= values['conf_thres2']/100) 

        #task1(model,size,conf)
        #task2(model,size,conf) 


        ### threading

        # task_thread1 = threading.Thread(target=program_camera1_FH_test, args=(model1, 416, values['conf_thres1']/100,))
        # task_thread2 = threading.Thread(target=program_camera2_FH_test, args=(model2, 416, values['conf_thres2']/100,))
        # task_thread3 = threading.Thread(target=program_camera3_FH_test, args=(model3, 416, values['conf_thres3']/100,))
 
        # task_thread1.start()
        # task_thread2.start()
        # task_thread3.start()
       


        #task_thread1 = threading.Thread(target=abc1)
        #task_thread2 = threading.Thread(target=abc2)

        #task_thread1.setDaemon(True)
        #task_thread2.setDaemon(True)

        #task_thread1.join()
        #task_thread2.join()


        # task_thread1 = multiprocessing.Process(target=program_camera1_FH_test, args=(model1, 416, values['conf_thres1']/100,))
        # task_thread2 = multiprocessing.Process(target=program_camera2_FH_test, args=(model2, 416, values['conf_thres2']/100,))
        # task_thread3 = multiprocessing.Process(target=program_camera3_FH_test, args=(model3, 416, values['conf_thres3']/100,))

        # task_thread1.start()
        # task_thread2.start()
        # task_thread3.start()
      

        # task_thread1.join()
        # task_thread2.join()
        # task_thread3.join()




        if event == 'check_model1' and values['check_model1'] == True:
            
            directory1 = 'E:/X75/BOMAU/DEN/CAM1/'
            print(directory1)
            if os.listdir(directory1) == []:
                print('folder 1 empty')
            else:
                print('received folder 1')
                cnt=0
                bomau = glob('E:/X75/BOMAU/DEN/CAM1/*.jpg')
                for path1 in bomau:
                    name = path1[9:]
                    cv2.waitKey(60)
                    check_ok=0
                    cnt += 1
                    #if (cnt > 14): cnt =1
                        
                    img1_orgin = cv2.imread(path1)


                    #img1_orgin = cv2.resize(img1_orgin,(640,480))  

                    img1_orgin = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB)     


                    result1 = model1(img1_orgin,size= 416,conf = values['conf_thres1']/100)

                    table1 = result1.pandas().xyxy[0]

                    area_remove1 = []

                    myresult1 =0 

                    for item in range(len(table1.index)):
                        width1 = table1['xmax'][item] - table1['xmin'][item]
                        height1 = table1['ymax'][item] - table1['ymin'][item]
                        #area1 = width1*height1
                        conf1 = table1['confidence'][item]
                        label_name = table1['name'][item]
                        for i1 in range(len(model1.names)):
                            if values[f'{model1.names[i1]}_1'] == True:
                                #if values[f'{model1.names[i1]}_WH'] == True:
                                if label_name == model1.names[i1]:
                                    if width1 < int(values[f'{model1.names[i1]}_Wn_1']): 
                                        table1.drop(item, axis=0, inplace=True)
                                        area_remove1.append(item)
                                    elif width1 > int(values[f'{model1.names[i1]}_Wx_1']): 
                                        table1.drop(item, axis=0, inplace=True)
                                        area_remove1.append(item)
                                    elif height1 < int(values[f'{model1.names[i1]}_Hn_1']): 
                                        table1.drop(item, axis=0, inplace=True)
                                        area_remove1.append(item)
                                    elif height1 > int(values[f'{model1.names[i1]}_Hx_1']): 
                                        table1.drop(item, axis=0, inplace=True)
                                        area_remove1.append(item)
                                    elif conf1 < int(values[f'{model1.names[i1]}_conf_1'])/100: 
                                        table1.drop(item, axis=0, inplace=True)
                                        area_remove1.append(item)
                            if values[f'{model1.names[i1]}_1'] == False:
                                if label_name == model1.names[i1]:
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)


                    names1 = list(table1['name'])

                    show1 = np.squeeze(result1.render(area_remove1))
                    show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                    show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB) 
                    #ta = time.time()
                    for i1 in range(len(model1.names)):
                        #register_ng = (3002 + i1*2).to_bytes(2, byteorder='big') + b'\x00'

                        if values[f'{model1.names[i1]}_OK_1'] == True:
                            len_name1 = 0
                            for name1 in names1:
                                if name1 == model1.names[i1]:
                                    len_name1 +=1
                            if len_name1 != int(values[f'{model1.names[i1]}_Num_1']):
                                print(cnt, 'NG', model1.names[i1])
                                #plc.write_word('D',register_ng,1)
                                cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                                window['result_cam1'].update(value= 'NG', text_color='red')
                                myresult1 = 1
                                

                        elif values[f'{model1.names[i1]}_NG_1'] == True:
                            if model1.names[i1] in names1:
                                print(cnt, 'NG', model1.names[i1])
                                #plc.write_word('D',register_ng,1)
                                cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                                window['result_cam1'].update(value= 'NG', text_color='red')    
                                myresult1 = 1         
                                    
                    cv2.putText(show1, str(cnt),(20,80),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),2)
                    if myresult1 == 0:
                        print(cnt,'OK')
                        check_ok = 1
                        #plc.write_word('D',(3000).to_bytes(2, byteorder='big') + b'\x00',1)
                        cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam1'].update(value= 'OK', text_color='green')

                    imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                    window['image1'].update(data= imgbytes1)
                    time.sleep(1)
                    if check_ok == 1:
                        if cnt<len(bomau):
                            sg.popup('Kiem mau chua dat')
                            break
                        else:
                            sg.popup('Kiem mau thanh cong')

        # thu mau Chau dien
        if event == 'check_model2' and values['check_model2'] == True and values['Tay_choi'] == False:
            check_ok=0
            directory2 =  'E:/X75/BOMAU/DEN/CD/'
            if os.listdir(directory2) == []:
                print('folder 2 empty')
            else:
                print('received folder 2')
                cnt = 0
                bomau = glob('E:/X75/BOMAU/DEN/CD/*')
                for path2 in bomau:
                    name = path2[9:]
                    cnt +=1
                    # if cnt >28: cnt =1
                    # stt = int(cnt/2) if cnt % 2 == 0 else int((cnt - 1 )/2) + 1
                    img2_orgin = cv2.imread(path2)
                    #img2_orgin = cv2.resize(img2_orgin,(640,480))  

                    img2_orgin = cv2.cvtColor(img2_orgin, cv2.COLOR_BGR2RGB) 

                    result2 = model2(img2_orgin,size= 416,conf = values['conf_thres2']/100)

                    table2 = result2.pandas().xyxy[0]

                    area_remove2 = []

                    myresult2 =0 

                    for item in range(len(table2.index)):
                        width2 = table2['xmax'][item] - table2['xmin'][item]
                        height2 = table2['ymax'][item] - table2['ymin'][item]
                        #area2 = width2*height2
                        label_name = table2['name'][item]
                        for i2 in range(len(model2.names)):
                            if values[f'{model2.names[i2]}_2'] == True:
                                #if values[f'{model2.names[i2]}_WH'] == True:
                                if label_name == model2.names[i2]:
                                    if width2 < int(values[f'{model2.names[i2]}_Wn_2']): 
                                        table2.drop(item, axis=0, inplace=True)
                                        area_remove2.append(item)
                                    elif width2 > int(values[f'{model2.names[i2]}_Wx_2']): 
                                        table2.drop(item, axis=0, inplace=True)
                                        area_remove2.append(item)
                                    elif height2 < int(values[f'{model2.names[i2]}_Hn_2']): 
                                        table2.drop(item, axis=0, inplace=True)
                                        area_remove2.append(item)
                                    elif height2 > int(values[f'{model2.names[i2]}_Hx_2']): 
                                        table2.drop(item, axis=0, inplace=True)
                                        area_remove2.append(item)
                        if values[f'{model2.names[i2]}_2'] == False:
                            if label_name == model2.names[i2]:
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                    names2 = list(table2['name'])

                    show2 = np.squeeze(result2.render(area_remove2))
                    show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                    show2 = cv2.cvtColor(show2, cv2.COLOR_BGR2RGB) 
                    #ta = time.time()
                    for i2 in range(len(model2.names)):
                        #register_ng = (4002 + i1*2).to_bytes(2, byteorder='big') + b'\x00'
                        if values[f'{model2.names[i2]}_OK_2'] == True:
                            len_name2 = 0
                            for name2 in names2:
                                if name2 == model2.names[i2]:
                                    len_name2 +=1
                            if len_name2 != int(values[f'{model2.names[i2]}_Num_2']):
                                print('NG')
                                #plc.write_word('D',register_ng,1)
                                cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                                window['result_cam2'].update(value= 'NG', text_color='red')
                                myresult2 = 1
                                break

                        if values[f'{model2.names[i2]}_NG_2'] == True:
                            if model2.names[i2] in names2:
                                print('NG')
                                #plc.write_word('D',register_ng,1)
                                cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                                window['result_cam2'].update(value= 'NG', text_color='red')    
                                myresult2 = 1      
                                break    
                    cv2.putText(show2, str(cnt),(20,80),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),2)
                    if myresult2 == 0:
                        print('OK')
                        check_ok = 1
                        #plc.write_word('D',(4000).to_bytes(2, byteorder='big') + b'\x00',1)
                        cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam2'].update(value= 'OK', text_color='green')

                    imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                    window['image2'].update(data= imgbytes2)
                    time.sleep(1)
                    if check_ok == 1:
                        if cnt<len(bomau):
                            sg.popup('Kiem mau chua dat')
                            break
                        else:
                            sg.popup('Kiem mau thanh cong')



        # thu mau Tay choi
        if event == 'check_model2' and values['check_model2'] == True and values['Tay_choi'] == True:
            check_ok=0
            directory2 = 'E:/TEST/TRANG//BUICHI'
            if os.listdir(directory2) == []:
                print('folder 2 empty')
            else:
                print('received folder 2')
                cnt = 0
                bomau = glob('E:/TEST/TRANG//BUICHI/*.JPG')
                for path2 in bomau:
                    ten = os.path.basename(path2)
                    cnt +=1
                    #if cnt >28: cnt =1
                    #stt = int(cnt/2) if cnt % 2 == 0 else int((cnt - 1 )/2) + 1
                    img2_orgin = cv2.imread(path2)
                    #img2_orgin = cv2.resize(img2_orgin,(640,480))  

                    img2_orgin = cv2.cvtColor(img2_orgin, cv2.COLOR_BGR2RGB) 

                    result2 = model3(img2_orgin,size= 416,conf = values['conf_thres3']/100)

                    table2 = result2.pandas().xyxy[0]

                    area_remove2 = []

                    myresult2 =0 

                    for item in range(len(table2.index)):
                        width2 = table2['xmax'][item] - table2['xmin'][item]
                        height2 = table2['ymax'][item] - table2['ymin'][item]
                        #area2 = width2*height2
                        label_name = table2['name'][item]
                        for i2 in range(len(model3.names)):
                            if values[f'{model3.names[i2]}_3'] == True:
                                #if values[f'{model3.names[i2]}_WH'] == True:
                                if label_name == model3.names[i2]:
                                    if width2 < int(values[f'{model3.names[i2]}_Wn_3']): 
                                        table2.drop(item, axis=0, inplace=True)
                                        area_remove2.append(item)
                                    elif width2 > int(values[f'{model3.names[i2]}_Wx_3']): 
                                        table2.drop(item, axis=0, inplace=True)
                                        area_remove2.append(item)
                                    elif height2 < int(values[f'{model3.names[i2]}_Hn_3']): 
                                        table2.drop(item, axis=0, inplace=True)
                                        area_remove2.append(item)
                                    elif height2 > int(values[f'{model3.names[i2]}_Hx_3']): 
                                        table2.drop(item, axis=0, inplace=True)
                                        area_remove2.append(item)
                        if values[f'{model3.names[i2]}_3'] == False:
                            if label_name == model3.names[i2]:
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                    names2 = list(table2['name'])

                    show2 = np.squeeze(result2.render(area_remove2))
                    show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                    show2 = cv2.cvtColor(show2, cv2.COLOR_BGR2RGB) 
                    #ta = time.time()
                    for i2 in range(len(model3.names)):
                        #register_ng = (4002 + i1*2).to_bytes(2, byteorder='big') + b'\x00'
                        if values[f'{model3.names[i2]}_OK_3'] == True:
                            len_name2 = 0
                            for name2 in names2:
                                if name2 == model3.names[i2]:
                                    len_name2 +=1
                            if len_name2 != int(values[f'{model3.names[i2]}_Num_3']):
                                print('NG')
                                #plc.write_word('D',register_ng,1)
                                #cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),3)
                                window['result_cam2'].update(value= 'NG', text_color='red')
                                myresult2 = 1
                                break

                        if values[f'{model3.names[i2]}_NG_3'] == True:
                            if model3.names[i2] in names2:
                                print('NG')
                                #plc.write_word('D',register_ng,1)
                                #cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),3)
                                window['result_cam2'].update(value= 'NG', text_color='red')    
                                myresult2 = 1      
                                break    
                    cv2.putText(show2, ten,(20,80),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
                    if myresult2 == 0:
                        print('OK')
                        check_ok = 1
                        #plc.write_word('D',(4000).to_bytes(2, byteorder='big') + b'\x00',1)
                        #cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam2'].update(value= 'OK', text_color='green')

                    imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                    window['image2'].update(data= imgbytes2)
                    #time.sleep(1)
                    if len(bomau)==cnt:
                        sg.popup('Het')
                    # if check_ok == 1:
                    #     if cnt<len(bomau):
                    #         sg.popup('Kiem mau chua dat')
                    #         break
                    #     else:
                    #         sg.popup('Kiem mau thanh cong')

        if event == 'Webcam1':
            #cap1 = cv2.VideoCapture(0)
            recording1 = True


        elif event == 'Stop1':
            recording1 = False 
            imgbytes1 = np.zeros([100,100,3],dtype=np.uint8)
            imgbytes1 = cv2.resize(imgbytes1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
            imgbytes1 = cv2.imencode('.png',imgbytes1)[1].tobytes()
            window['image1'].update(data=imgbytes1)
            window['result_cam1'].update(value='')


        if event == 'Webcam2':
            #cap2 = cv2.VideoCapture(1)
            recording2 = True


        elif event == 'Stop2':
            recording2 = False 
            imgbytes2 = np.zeros([100,100,3],dtype=np.uint8)
            imgbytes2 = cv2.resize(imgbytes2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
            imgbytes2 = cv2.imencode('.png',imgbytes2)[1].tobytes()
            window['image2'].update(data=imgbytes2)
            window['result_cam2'].update(value='')


        #if recording1:
            # if values['have_model1'] == True:
            #     img1_orgin = my_callback1.image 
            #     img1_orgin = img1_orgin[50:530,70:710]
            #     img1_orgin = img1_orgin.copy()
            #     img1_orgin = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB)                              
            #     result1 = model1(img1_orgin,size= 416,conf= values['conf_thres1']/100)             
            #     table1 = result1.pandas().xyxy[0]
            #     area_remove1 = []

            #     myresult1 =0 

            #     for item in range(len(table1.index)):
            #         width1 = table1['xmax'][item] - table1['xmin'][item]
            #         height1 = table1['ymax'][item] - table1['ymin'][item]
            #         #area1 = width1*height1
            #         label_name = table1['name'][item]
            #         for i1 in range(len(model1.names)):
            #             if values[f'{model1.names[i1]}_1'] == True:
            #                 #if values[f'{model1.names[i1]}_WH'] == True:
            #                 if label_name == model1.names[i1]:
            #                     if width1 < int(values[f'{model1.names[i1]}_Wn_1']): 
            #                         table1.drop(item, axis=0, inplace=True)
            #                         area_remove1.append(item)
            #                     elif width1 > int(values[f'{model1.names[i1]}_Wx_1']): 
            #                         table1.drop(item, axis=0, inplace=True)
            #                         area_remove1.append(item)
            #                     elif height1 < int(values[f'{model1.names[i1]}_Hn_1']): 
            #                         table1.drop(item, axis=0, inplace=True)
            #                         area_remove1.append(item)
            #                     elif height1 > int(values[f'{model1.names[i1]}_Hx_1']): 
            #                         table1.drop(item, axis=0, inplace=True)
            #                         area_remove1.append(item)

            #     names1 = list(table1['name'])

            #     show1 = np.squeeze(result1.render(area_remove1))
            #     show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
        
            #     #ta = time.time()
            #     for i1 in range(len(model1.names)):
            #         if values[f'{model1.names[i1]}_OK_1'] == True:
            #             len_name1 = 0
            #             for name1 in names1:
            #                 if name1 == model1.names[i1]:
            #                     len_name1 +=1
            #             if len_name1 != int(values[f'{model1.names[i1]}_Num_1']):
            #                 print('NG')
            #                 cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
            #                 window['result_cam1'].update(value= 'NG', text_color='red')
            #                 myresult1 = 1
            #                 break

            #         if values[f'{model1.names[i1]}_NG_1'] == True:
            #             if model1.names[i1] in names1:
            #                 print('NG')
            #                 cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
            #                 window['result_cam1'].update(value= 'NG', text_color='red')    
            #                 myresult1 = 1         
            #                 break    

            #     if myresult1 == 0:
            #         print('OK')
            #         cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
            #         window['result_cam1'].update(value= 'OK', text_color='green')
                
            #     imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
            #     window['image1'].update(data= imgbytes1)
            # else:
            #img1_orgin = my_callback1.image 
            #img1_orgin = img1_orgin[50:530,70:710]
            #img1_orgin = img1_orgin.copy()
            #img1_orgin = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB) 
            #img1_resize = cv2.resize(img1_orgin,(image_width_display,image_height_display))
            # if img1_orgin is not None:
            #     show1 = img1_resize
            #     imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
            #     window['image1'].update(data=imgbytes1)
            #     window['result_cam1'].update(value='')


        #if recording2:
            # if values['have_model2'] == True:
            #     img2_orgin = my_callback2.image  
            #     img2_orgin = img2_orgin[50:530,70:710]
            #     img2_orgin = img2_orgin.copy()
            #     img2_orgin = cv2.cvtColor(img2_orgin, cv2.COLOR_BGR2RGB)                              
            #     result2 = model2(img2_orgin,size= 416,conf= values['conf_thres2']/100)             
            #     table2 = result2.pandas().xyxy[0]
            #     area_remove2 = []

            #     myresult2 =0 

            #     for item in range(len(table2.index)):
            #         width2 = table2['xmax'][item] - table2['xmin'][item]
            #         height2 = table2['ymax'][item] - table2['ymin'][item]
            #         #area2 = width2*height2
            #         label_name = table2['name'][item]
            #         for i2 in range(len(model2.names)):
            #             if values[f'{model2.names[i2]}_2'] == True:
            #                 if label_name == model2.names[i2]:
            #                     if width2 < int(values[f'{model2.names[i2]}_Wn_2']): 
            #                         table2.drop(item, axis=0, inplace=True)
            #                         area_remove2.append(item)
            #                     elif width2 > int(values[f'{model2.names[i2]}_Wx_2']): 
            #                         table2.drop(item, axis=0, inplace=True)
            #                         area_remove2.append(item)
            #                     elif height2 < int(values[f'{model2.names[i2]}_Hn_2']): 
            #                         table2.drop(item, axis=0, inplace=True)
            #                         area_remove2.append(item)
            #                     elif height2 > int(values[f'{model2.names[i2]}_Hx_2']): 
            #                         table2.drop(item, axis=0, inplace=True)
            #                         area_remove2.append(item)

            #     names2 = list(table2['name'])

            #     show2 = np.squeeze(result2.render(area_remove2))
            #     show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
        
            #     #ta = time.time()
            #     for i2 in range(len(model2.names)):
            #         if values[f'{model2.names[i2]}_OK_2'] == True:
            #             len_name2 = 0
            #             for name2 in names2:
            #                 if name2 == model2.names[i2]:
            #                     len_name2 +=2
            #             if len_name2 != int(values[f'{model2.names[i2]}_Num_2']):
            #                 print('NG')
            #                 cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
            #                 window['result_cam2'].update(value= 'NG', text_color='red')
            #                 myresult2 = 1
            #                 break

            #         if values[f'{model2.names[i2]}_NG_2'] == True:
            #             if model2.names[i2] in names2:
            #                 print('NG')
            #                 cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
            #                 window['result_cam2'].update(value= 'NG', text_color='red')    
            #                 myresult2 = 1         
            #                 break    

            #     if myresult2 == 0:
            #         print('OK')
            #         cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
            #         window['result_cam2'].update(value= 'OK', text_color='green')
                
            #     imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
            #     window['image2'].update(data= imgbytes2)
            # else:
            # img2_orgin = my_callback2.image  
            # img2_orgin = img2_orgin[50:530,70:710]
            # img2_orgin = img2_orgin.copy()
            # img2_orgin = cv2.cvtColor(img2_orgin, cv2.COLOR_BGR2RGB) 
            # img2_resize = cv2.resize(img2_orgin,(image_width_display,image_height_display))
            # if img2_orgin is not None:
            #     show2 = img2_resize
            #     imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
            #     window['image2'].update(data=imgbytes2)
            #     window['result_cam2'].update(value='')


        if event == 'Pic1':
            dir_img1 = sg.popup_get_file('Choose your image 1',file_types=file_name_img,keep_on_top= True)
            if dir_img1 not in ('',None):
                pic1 = Image.open(dir_img1)
                img1_resize = pic1.resize((image_width_display,image_height_display))
                imgbytes1 = ImageTk.PhotoImage(img1_resize)
                window['image1'].update(data= imgbytes1)
                window['Detect1'].update(disabled= False)         

        if event == 'Pic2':
            dir_img2 = sg.popup_get_file('Choose your image 2',file_types=file_name_img,keep_on_top= True)
            if dir_img2 not in ('',None):
                pic2 = Image.open(dir_img2)
                img2_resize = pic2.resize((image_width_display,image_height_display))
                imgbytes2 = ImageTk.PhotoImage(img2_resize)
                window['image2'].update(data=imgbytes2)
                window['Detect2'].update(disabled= False)


        if event == 'Change1' or event == 'Change_1':
            mypath1 = values['file_weights1']
            model1= torch.hub.load('./levu','custom',path=mypath1,source='local',force_reload=True)
            mychoose = values['choose_model']
            
            weight1 = values['file_weights1']
            conf_thres1 = values['conf_thres1'] 
            OK_Cam1 = values['have_save_OK_1']
            NG_Cam1 = values['have_save_NG_1']
            Folder_OK_Cam1 = values['save_OK_1']
            Folder_NG_Cam1 = values['save_NG_1']
            
            weight2 = values['file_weights2']            
            conf_thres2 = values['conf_thres2'] 
            OK_Cam2 = values['have_save_OK_2']
            NG_Cam2 = values['have_save_NG_2']
            Folder_OK_Cam2 = values['save_OK_2']
            Folder_NG_Cam2 = values['save_NG_2']

            weight3 = values['file_weights3']
            conf_thres3 = values['conf_thres3'] 
            OK_Cam3 = values['have_save_OK_3']         
            NG_Cam3 = values['have_save_NG_3']          
            Folder_OK_Cam3 = values['save_OK_3']           
            Folder_NG_Cam3 = values['save_NG_3']   
            
            window.close() 
            window = make_window(theme)

            window['choose_model'].update(value=mychoose)
            
            window['file_weights1'].update(value=weight1)
            window['conf_thres1'].update(value=conf_thres1)
            window['have_save_OK_1'].update(value=OK_Cam1)
            window['have_save_NG_1'].update(value=NG_Cam1)
            window['save_OK_1'].update(value=Folder_OK_Cam1)
            window['save_NG_1'].update(value=Folder_NG_Cam1)
            
            window['file_weights2'].update(value=weight2)
            window['conf_thres2'].update(value=conf_thres2)
            window['have_save_OK_2'].update(value=OK_Cam2)
            window['have_save_NG_2'].update(value=NG_Cam2)
            window['save_OK_2'].update(value=Folder_OK_Cam2)
            window['save_NG_2'].update(value=Folder_NG_Cam2)

            window['file_weights3'].update(value=weight3)
            window['conf_thres3'].update(value=conf_thres3)
            window['have_save_OK_3'].update(value=OK_Cam3)  
            window['have_save_NG_3'].update(value=NG_Cam3)
            window['save_OK_3'].update(value=Folder_OK_Cam3)
            window['save_NG_3'].update(value=Folder_NG_Cam3)
           
           

        if event == 'Change2' or event == 'Change_2':
            mypath2 = values['file_weights2']
            model2= torch.hub.load('./levu','custom',path=mypath2,source='local',force_reload=False)
            mychoose = values['choose_model']
            
            weight1 = values['file_weights1']
            conf_thres1 = values['conf_thres1'] 
            OK_Cam1 = values['have_save_OK_1']
            NG_Cam1 = values['have_save_NG_1']
            Folder_OK_Cam1 = values['save_OK_1']
            Folder_NG_Cam1 = values['save_NG_1']

            weight2 = values['file_weights2']            
            conf_thres2 = values['conf_thres2'] 
            OK_Cam2 = values['have_save_OK_2']
            NG_Cam2 = values['have_save_NG_2']
            Folder_OK_Cam2 = values['save_OK_2']
            Folder_NG_Cam2 = values['save_NG_2']

            weight3 = values['file_weights3']
            conf_thres3 = values['conf_thres3'] 
            OK_Cam3 = values['have_save_OK_3']         
            NG_Cam3 = values['have_save_NG_3']          
            Folder_OK_Cam3 = values['save_OK_3']           
            Folder_NG_Cam3 = values['save_NG_3']       


            window.close() 
            window = make_window(theme)

            window['choose_model'].update(value=mychoose)
            
            window['file_weights1'].update(value=weight1)
            window['conf_thres1'].update(value=conf_thres1)
            window['have_save_OK_1'].update(value=OK_Cam1)
            window['have_save_NG_1'].update(value=NG_Cam1)
            window['save_OK_1'].update(value=Folder_OK_Cam1)
            window['save_NG_1'].update(value=Folder_NG_Cam1)

            window['file_weights2'].update(value=weight2)
            window['conf_thres2'].update(value=conf_thres2)
            window['have_save_OK_2'].update(value=OK_Cam2)
            window['have_save_NG_2'].update(value=NG_Cam2)
            window['save_OK_2'].update(value=Folder_OK_Cam2)
            window['save_NG_2'].update(value=Folder_NG_Cam2)

            window['file_weights3'].update(value=weight3)
            window['conf_thres3'].update(value=conf_thres3)
            window['have_save_OK_3'].update(value=OK_Cam3)  
            window['have_save_NG_3'].update(value=NG_Cam3)
            window['save_OK_3'].update(value=Folder_OK_Cam3)
            window['save_NG_3'].update(value=Folder_NG_Cam3)


        if event == 'Change_3':
            mypath3 = values['file_weights3']
            model3= torch.hub.load('./levu','custom',path=mypath3,source='local',force_reload=False)
            mychoose = values['choose_model']

            weight1 = values['file_weights1']
            conf_thres1 = values['conf_thres1'] 
            OK_Cam1 = values['have_save_OK_1']
            NG_Cam1 = values['have_save_NG_1']
            Folder_OK_Cam1 = values['save_OK_1']
            Folder_NG_Cam1 = values['save_NG_1']

            weight2 = values['file_weights2']            
            conf_thres2 = values['conf_thres2'] 
            OK_Cam2 = values['have_save_OK_2']
            NG_Cam2 = values['have_save_NG_2']
            Folder_OK_Cam2 = values['save_OK_2']
            Folder_NG_Cam2 = values['save_NG_2']

            weight3 = values['file_weights3']
            conf_thres3 = values['conf_thres3'] 
            OK_Cam3 = values['have_save_OK_3']         
            NG_Cam3 = values['have_save_NG_3']          
            Folder_OK_Cam3 = values['save_OK_3']           
            Folder_NG_Cam3 = values['save_NG_3']
           

            window.close() 
            window = make_window(theme)

            window['choose_model'].update(value=mychoose)
            
            window['file_weights1'].update(value=weight1)
            window['conf_thres1'].update(value=conf_thres1)
            window['have_save_OK_1'].update(value=OK_Cam1)
            window['have_save_NG_1'].update(value=NG_Cam1)
            window['save_OK_1'].update(value=Folder_OK_Cam1)
            window['save_NG_1'].update(value=Folder_NG_Cam1)

            window['file_weights2'].update(value=weight2)
            window['conf_thres2'].update(value=conf_thres2)
            window['have_save_OK_2'].update(value=OK_Cam2)
            window['have_save_NG_2'].update(value=NG_Cam2)
            window['save_OK_2'].update(value=Folder_OK_Cam2)
            window['save_NG_2'].update(value=Folder_NG_Cam2)

            window['file_weights3'].update(value=weight3)
            window['conf_thres3'].update(value=conf_thres3)
            window['have_save_OK_3'].update(value=OK_Cam3)  
            window['have_save_NG_3'].update(value=NG_Cam3)
            window['save_OK_3'].update(value=Folder_OK_Cam3)
            window['save_NG_3'].update(value=Folder_NG_Cam3)


        if event == 'Detect1':
            print('CAM 1 DETECT')
            t1 = time.time()
            try:
            
                result1 = model1(pic1,size= 416,conf = values['conf_thres1']/100)

                table1 = result1.pandas().xyxy[0]
                print(table1)
                area_remove1 = []

                myresult1 =0 

                for item in range(len(table1.index)):
                    width1 = table1['xmax'][item] - table1['xmin'][item]
                    height1 = table1['ymax'][item] - table1['ymin'][item]
                    #area1 = width1*height1
                    conf1 = table1['confidence'][item]
                    label_name = table1['name'][item]
                    for i1 in range(len(model1.names)):
                        if values[f'{model1.names[i1]}_1'] == True:
                            #if values[f'{model1.names[i1]}_WH'] == True:
                            if label_name == model1.names[i1]:
                                if width1 < int(values[f'{model1.names[i1]}_Wn_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif width1 > int(values[f'{model1.names[i1]}_Wx_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif height1 < int(values[f'{model1.names[i1]}_Hn_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif height1 > int(values[f'{model1.names[i1]}_Hx_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif conf1 < int(values[f'{model1.names[i1]}_conf_1'])/100: 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                        if values[f'{model1.names[i1]}_1'] == False:
                            if label_name == model1.names[i1]:
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                names1 = list(table1['name'])

                show1 = np.squeeze(result1.render(area_remove1))
                show_1 = cv2.resize(show1, (720,540), interpolation = cv2.INTER_AREA)
                show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)
                show_1 = cv2.cvtColor(show_1, cv2.COLOR_BGR2RGB)
                #ta = time.time()
                k=1
                for i1 in range(len(model1.names)):
                    if values[f'{model1.names[i1]}_1'] == True:
                        if values[f'{model1.names[i1]}_OK_1'] == True:
                            len_name1 = 0
                            for name1 in names1:
                                if name1 == model1.names[i1]:
                                    len_name1 +=1
                            if len_name1 != int(values[f'{model1.names[i1]}_Num_1']):
                                print('NG')
                                cv2.putText(show1,model1.names[i1],(30,50*k),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),0)
                                print(model1.names[i1])
                                window['result_cam1'].update(value= 'NG', text_color='red')
                                myresult1 = 1
                                k+=1
                                break

                        if values[f'{model1.names[i1]}_NG_1'] == True:
                            if model1.names[i1] in names1:
                                print('NG')
                                cv2.putText(show1,model1.names[i1],(30,50*k),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),0)
                                print(model1.names[i1])
                                window['result_cam1'].update(value= 'NG', text_color='red')    
                                myresult1 = 1   
                                k+=1      
                                break    

                if myresult1 == 0:
                    print('OK')
                    cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
                    cv2.putText(show_1, 'OK',(1000,100),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
                    window['result_cam1'].update(value= 'OK', text_color='green')
                else:
                    cv2.putText(show1,'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                    
                imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                window['image1'].update(data= imgbytes1)
                                
                imgbytes1 = cv2.imencode('.png',show_1)[1].tobytes()
                window['toan1'].update(data= imgbytes1)
                
            
            except:
                print(traceback.format_exc())
                sg.popup_annoying("Don't have image or parameter wrong", font=('Helvetica',14),text_color='red')
            
            t2 = time.time() - t1
            print(t2)
            time_cam1 = str(int(t2*1000)) + 'ms'
            window['time_cam1'].update(value= time_cam1, text_color='black') 
            print('---------------------------------------------') 


            
        if event == 'Detect2' and values['Tay_choi'] == False:
            print('Chau dien')
            t1 = time.time()
            try:
    
                result2 = model2(pic2,size= 416,conf = values['conf_thres2']/100)

                table2 = result2.pandas().xyxy[0]

                area_remove2 = []

                myresult2 =0 

                for item in range(len(table2.index)):
                    width2 = table2['xmax'][item] - table2['xmin'][item]
                    height2 = table2['ymax'][item] - table2['ymin'][item]
                    #area2 = width2*height2
                    label_name = table2['name'][item]
                    for i2 in range(len(model2.names)):
                        if values[f'{model2.names[i2]}_2'] == True:
                            #if values[f'{model2.names[i2]}_WH'] == True:
                            if label_name == model2.names[i2]:
                                if width2 < int(values[f'{model2.names[i2]}_Wn_2']): 
                                    table2.drop(item, axis=0, inplace=True)
                                    area_remove2.append(item)
                                elif width2 > int(values[f'{model2.names[i2]}_Wx_2']): 
                                    table2.drop(item, axis=0, inplace=True)
                                    area_remove2.append(item)
                                elif height2 < int(values[f'{model2.names[i2]}_Hn_2']): 
                                    table2.drop(item, axis=0, inplace=True)
                                    area_remove2.append(item)
                                elif height2 > int(values[f'{model2.names[i2]}_Hx_2']): 
                                    table2.drop(item, axis=0, inplace=True)
                                    area_remove2.append(item)

                        if values[f'{model2.names[i2]}_2'] == False:
                            if label_name == model2.names[i2]:
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                names2 = list(table2['name'])

                show2 = np.squeeze(result2.render(area_remove2))
                show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                show2 = cv2.cvtColor(show2, cv2.COLOR_BGR2RGB)
                #ta = time.time()
                for i2 in range(len(model2.names)):
                    if values[f'{model2.names[i2]}_OK_2'] == True:
                        len_name2 = 0
                        for name2 in names2:
                            if name2 == model2.names[i2]:
                                len_name2 +=1
                        if len_name2 != int(values[f'{model2.names[i2]}_Num_2']):
                            print('NG')
                            cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                            window['result_cam2'].update(value= 'NG', text_color='red')
                            myresult2 = 1
                            break

                    if values[f'{model2.names[i2]}_NG_2'] == True:
                        if model2.names[i2] in names2:
                            print('NG')
                            cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                            window['result_cam2'].update(value= 'NG', text_color='red')    
                            myresult2 = 1      
                            break    

                if myresult2 == 0:
                    print('OK')
                    cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
                    window['result_cam2'].update(value= 'OK', text_color='green')

                imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                window['image2'].update(data= imgbytes2)

            
            except:
                print(traceback.format_exc())
                sg.popup_annoying("Don't have image or parameter wrong", font=('Helvetica',24),text_color='red')
            
            t2 = time.time() - t1
            print(t2)
            time_cam2 = str(int(t2*1000)) + 'ms'
            window['time_cam2'].update(value= time_cam2, text_color='black') 
            print('---------------------------------------------') 



        if event == 'Detect2' and values['Tay_choi'] == True:
            print('Tay choi')
            t1 = time.time()
            try:
                result3 = model3(pic2,size= 416,conf = values['conf_thres3']/100)

                table3 = result3.pandas().xyxy[0]

                area_remove3 = []

                myresult3 =0 

                for item in range(len(table3.index)):
                    width3 = table3['xmax'][item] - table3['xmin'][item]
                    height3 = table3['ymax'][item] - table3['ymin'][item]
                    #area3 = width3*height3
                    label_name = table3['name'][item]
                    for i3 in range(len(model3.names)):
                        if values[f'{model3.names[i3]}_3'] == True:
                            #if values[f'{model3.names[i3]}_WH'] == True:
                            if label_name == model3.names[i3]:
                                if width3 < int(values[f'{model3.names[i3]}_Wn_3']): 
                                    table3.drop(item, axis=0, inplace=True)
                                    area_remove3.append(item)
                                elif width3 > int(values[f'{model3.names[i3]}_Wx_3']): 
                                    table3.drop(item, axis=0, inplace=True)
                                    area_remove3.append(item)
                                elif height3 < int(values[f'{model3.names[i3]}_Hn_3']): 
                                    table3.drop(item, axis=0, inplace=True)
                                    area_remove3.append(item)
                                elif height3 > int(values[f'{model3.names[i3]}_Hx_3']): 
                                    table3.drop(item, axis=0, inplace=True)
                                    area_remove3.append(item)
                        if values[f'{model3.names[i3]}_3'] == False:
                            if label_name == model3.names[i3]:
                                table3.drop(item, axis=0, inplace=True)
                                area_remove3.append(item)

                names3 = list(table3['name'])

                show3 = np.squeeze(result3.render(area_remove3))
                show3 = cv2.resize(show3, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                show3 = cv2.cvtColor(show3, cv2.COLOR_BGR2RGB)
                #ta = time.time()
                for i3 in range(len(model3.names)):
                    if values[f'{model3.names[i3]}_OK_3'] == True:
                        len_name3 = 0
                        for name3 in names3:
                            if name3 == model3.names[i3]:
                                len_name3 +=1
                        if len_name3 != int(values[f'{model3.names[i3]}_Num_3']):
                            print('NG')
                            cv2.putText(show3, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                            window['result_cam2'].update(value= 'NG', text_color='red')
                            myresult3 = 1
                            break

                    if values[f'{model3.names[i3]}_NG_3'] == True:
                        if model3.names[i3] in names3:
                            print('NG')
                            cv2.putText(show3, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                            window['result_cam2'].update(value= 'NG', text_color='red')    
                            myresult3 = 1         
                            break    

                if myresult3 == 0:
                    print('OK')
                    cv2.putText(show3, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
                    window['result_cam2'].update(value= 'OK', text_color='green')

                imgbytes3 = cv2.imencode('.png',show3)[1].tobytes()
                window['image2'].update(data= imgbytes3)

            
            except:
                print(traceback.format_exc())
                sg.popup_annoying("Don't have image or parameter wrong", font=('Helvetica',34),text_color='red')
            
            t2 = time.time() - t1
            print(t2)
            time_cam3 = str(int(t2*1000)) + 'ms'
            window['time_cam2'].update(value= time_cam3, text_color='black') 
            print('---------------------------------------------') 

    window.close() 

except Exception as e:
    #plc.write_word('D',490,0) 
    print(traceback.print_exc())
    str_error = str(e)    
    sg.popup(str_error,font=('Helvetica',15), text_color='red',keep_on_top= True)
              