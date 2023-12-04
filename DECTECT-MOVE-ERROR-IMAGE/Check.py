from copyreg import remove_extension
from faulthandler import disable
import glob
import os
from tkinter.tix import Tree
import cv2
import threading
import torch
import numpy as np 
import time

import PySimpleGUI as sg

from PIL import Image,ImageTk
import os
import datetime 
import shutil

from PIL import Image
from yaml import load

from udp import UDPFinsConnection
from initialization import FinsPLCMemoryAreas

import traceback

import sqlite3

#import stapipy as st
import multiprocessing
import keyboard


mysleep = 0.1

SCALE_X_CAM1 = 640*1.2/2048
SCALE_Y_CAM1 = 480*1.2/1536

SCALE_X_CAM2 = 640/1440
SCALE_Y_CAM2 = 480/1080



def removefile():
    directory1 = 'C:/FH/camera1/'

    if os.listdir(directory1) != []:
        for i in glob.glob(directory1+'*'):
            for j in glob.glob(i+'/*'):
                os.remove(j)
            os.rmdir(i)


    print('already delete folder')





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
        a += 8


def save_all(model,i):
    with open('static/all'+ str(i) + '.txt','w') as f:
        f.write('weights' + str(i) + '=' + str(values['file_weights' + str(i)]))
        f.write('\n')
        f.write('conf' + str(i) + '=' + str(values['conf_thres' + str(i)]))
        f.write('\n')

        for item in range(len(model.names)):
            f.write(str(f'{model.names[item]}_' + str(i)) + '=' + str(values[f'{model.names[item]}_' + str(i)]))
            f.write('\n')
            f.write(str(f'{model.names[item]}_OK_' + str(i)) + '=' + str(values[f'{model.names[item]}_OK_' + str(i)]))
            f.write('\n')
            f.write(str(f'{model.names[item]}_Num_' + str(i)) + '=' + str(values[f'{model.names[item]}_Num_' + str(i)]))
            f.write('\n')
            f.write(str(f'{model.names[item]}_NG_' + str(i)) + '=' + str(values[f'{model.names[item]}_NG_' + str(i)]))
            f.write('\n')
            f.write(str(f'{model.names[item]}_Wn_' + str(i)) + '=' + str(values[f'{model.names[item]}_Wn_' + str(i)]))
            f.write('\n')
            f.write(str(f'{model.names[item]}_Wx_' + str(i)) + '=' + str(values[f'{model.names[item]}_Wx_' + str(i)]))
            f.write('\n')
            f.write(str(f'{model.names[item]}_Hn_' + str(i)) + '=' + str(values[f'{model.names[item]}_Hn_' + str(i)]))
            f.write('\n')
            f.write(str(f'{model.names[item]}_Hx_' + str(i)) + '=' + str(values[f'{model.names[item]}_Hx_' + str(i)]))
            if item != len(model.names)-1:
                f.write('\n')



def load_all_sql(i,choose_model):
    conn = sqlite3.connect('modeldb_1_Check.db')
    cursor = conn.execute("SELECT ChooseModel,Camera,Weights,Confidence,have_1,have_2,have_3,have_4,have_5,have_6,have_7,have_8,have_9,have_10,folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10,Joined,Ok,Num,NG,WidthMin,WidthMax,HeightMin,HeightMax,PLC_NG,PLC_OK from MYMODEL")
    for row in cursor:
        #if row[0] == values['choose_model']:
        if row[0] == choose_model:
            row1_a, row1_b = row[1].strip().split('_')
            if row1_a == str(i) and row1_b == '0':
                window['file_weights' + str(i)].update(value=row[2])
                window['conf_thres' + str(i)].update(value=row[3])

                window['have_save_1'].update(value=str2bool(row[4]))
                window['have_save_2'].update(value=str2bool(row[5]))
                window['have_save_3'].update(value=str2bool(row[6]))
                window['have_save_4'].update(value=str2bool(row[7]))
                window['have_save_5'].update(value=str2bool(row[8]))
                window['have_save_6'].update(value=str2bool(row[9]))
                window['have_save_7'].update(value=str2bool(row[10]))
                window['have_save_8'].update(value=str2bool(row[11]))
                window['have_save_9'].update(value=str2bool(row[12]))
                window['have_save_10'].update(value=str2bool(row[13]))
            
                window['save_1'].update(value=row[14])
                window['save_2'].update(value=row[15])        
                window['save_3'].update(value=row[16])         
                window['save_4'].update(value=row[17])        
                window['save_5'].update(value=row[18])    
                window['save_6'].update(value=row[19])
                window['save_7'].update(value=row[20])        
                window['save_8'].update(value=row[21])        
                window['save_9'].update(value=row[22])        
                window['save_10'].update(value=row[23])
      



                model = torch.hub.load('./levu','custom', path= row[2], source='local',force_reload =False)
            if row1_a == str(i):
                for item in range(len(model.names)):
                    if int(row1_b) == item:
                        window[f'{model.names[item]}_' + str(i)].update(value=str2bool(row[24]))
                        window[f'{model.names[item]}_OK_' + str(i)].update(value=str2bool(row[25]))
                        window[f'{model.names[item]}_Num_' + str(i)].update(value=str(row[26]))
                        window[f'{model.names[item]}_NG_' + str(i)].update(value=str2bool(row[27]))
                        window[f'{model.names[item]}_Wn_' + str(i)].update(value=str(row[28]))
                        window[f'{model.names[item]}_Wx_' + str(i)].update(value=str(row[29]))
                        window[f'{model.names[item]}_Hn_' + str(i)].update(value=str(row[30]))
                        window[f'{model.names[item]}_Hx_' + str(i)].update(value=str(row[31]))
                        window[f'{model.names[item]}_PLC_' + str(i)].update(value=str(row[32]))
                        window[f'OK_PLC_' + str(i)].update(value=str(row[33]))

                    

    conn.close()


def save_all_sql(model,i,choose_model):
    conn = sqlite3.connect('modeldb_1_Check.db')
    cursor = conn.execute("SELECT ChooseModel,Camera,Weights,Confidence,have_1,have_2,have_3,have_4,have_5,have_6,have_7,have_8,have_9,have_10,folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10,Joined,Ok,Num,NG,WidthMin,WidthMax,HeightMin,HeightMax,PLC_NG,PLC_OK from MYMODEL")
    update = 0 

    for row in cursor:
        if row[0] == choose_model:            
            row1_a, _ = row[1].strip().split('_')
            if row1_a == str(i):
                conn.execute("DELETE FROM MYMODEL WHERE (ChooseModel = ? AND Camera LIKE ?)", (choose_model,str(i) + '%'))
                for item in range(len(model.names)):
                    #conn.execute("UPDATE MYMODEL SET ChooseModel = ? , Camera = ?, Weights = ?,Confidence = ?, Joined = ?, Ok = ?, Num = ?, NG = ?, WidthMin = ?, WidthMax = ?, HeightMin = ?, HeightMax = ? WHERE (ChooseModel = ? AND Camera = ?)",(str(values['choose_model']),str(i)+ '_' +str(item) ,str(values['file_weights' + str(i)]),int(values['conf_thres' + str(i)]), str(values[model.names[item] + '_' + str(i)]), str(values[model.names[item]+ '_OK_' + str(i)]), int(values[model.names[item]+ '_Num_' + str(i)]), str(values[model.names[item]+ '_NG_' + str(i)]), int(values[model.names[item] + '_Wn_' + str(i)]), int(values[model.names[item] + '_Wx_' + str(i)]), int(values[model.names[item]+ '_Hn_' + str(i)]), int(values[model.names[item] + '_Hx_' + str(i)]), choose_model,str(i) + '_' + str(item)))
                    #conn.execute("DELETE FROM MYMODEL WHERE (ChooseModel = ? AND Camera = ?)", (choose_model,str(i) + '_' + str(item)))
                    conn.execute("INSERT INTO MYMODEL (ChooseModel,Camera, Weights,Confidence,have_1,have_2,have_3,have_4,have_5,have_6,have_7,have_8,have_9,have_10,folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10,Joined,Ok,Num,NG,WidthMin, WidthMax,HeightMin,HeightMax,PLC_NG,PLC_OK) \
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (str(values['choose_model']),str(i)+ '_' +str(item) ,str(values['file_weights' + str(i)]), int(values['conf_thres' + str(i)]),str(values['have_save_1']),str(values['have_save_2']),str(values['have_save_3']),str(values['have_save_4']),str(values['have_save_5']),str(values['have_save_6']),str(values['have_save_7']),str(values['have_save_8']),str(values['have_save_9']),str(values['have_save_10']),str(values['save_1']),str(values['save_2']),str(values['save_3']),str(values['save_4']),str(values['save_5']),str(values['save_6']),str(values['save_7']),str(values['save_8']),str(values['save_9']),str(values['save_10']),str(values[f'{model.names[item]}_' + str(i)]), str(values[f'{model.names[item]}_OK_' + str(i)]), int(values[f'{model.names[item]}_Num_' + str(i)]), str(values[f'{model.names[item]}_NG_' + str(i)]), int(values[f'{model.names[item]}_Wn_' + str(i)]), int(values[f'{model.names[item]}_Wx_' + str(i)]), int(values[f'{model.names[item]}_Hn_' + str(i)]), int(values[f'{model.names[item]}_Hx_' + str(i)]), int(values[f'{model.names[item]}_PLC_' + str(i)]), int(values['OK_PLC_' + str(i)])))           
                    #conn.execute("INSERT INTO MYMODEL (ChooseModel,Camera, Weights,Confidence,Joined,Ok,Num,NG,WidthMin, WidthMax,HeightMin,HeightMax) \
                    #    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (str(values['choose_model']),str(i)+ '_' +str(item) ,str(values['file_weights' + str(i)]), int(values['conf_thres' + str(i)]),str(values[f'{model.names[item]}_' + str(i)]), str(values[f'{model.names[item]}_OK_' + str(i)]), int(values[f'{model.names[item]}_Num_' + str(i)]), str(values[f'{model.names[item]}_NG_' + str(i)]), int(values[f'{model.names[item]}_Wn_' + str(i)]), int(values[f'{model.names[item]}_Wx_' + str(i)]), int(values[f'{model.names[item]}_Hn_' + str(i)]), int(values[f'{model.names[item]}_Hx_' + str(i)])))           
                    update = 1
                break

    if update == 0:
        for item in range(len(model.names)):
            conn.execute("INSERT INTO MYMODEL (ChooseModel,Camera, Weights,Confidence,have_1,have_2,have_3,have_4,have_5,have_6,have_7,have_8,have_9,have_10,folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10,Joined,Ok,Num,NG,WidthMin, WidthMax,HeightMin,HeightMax,PLC_NG,PLC_OK) \
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (str(values['choose_model']),str(i)+ '_' +str(item) ,str(values['file_weights' + str(i)]), int(values['conf_thres' + str(i)]),str(values['have_save_1']),str(values['have_save_2']),str(values['have_save_3']),str(values['have_save_4']),str(values['have_save_5']),str(values['have_save_6']),str(values['have_save_7']),str(values['have_save_8']),str(values['have_save_9']),str(values['have_save_10']),str(values['save_1']),str(values['save_2']),str(values['save_3']),str(values['save_4']),str(values['save_5']),str(values['save_6']),str(values['save_7']),str(values['save_8']),str(values['save_9']),str(values['save_10']),str(values[f'{model.names[item]}_' + str(i)]), str(values[f'{model.names[item]}_OK_' + str(i)]), int(values[f'{model.names[item]}_Num_' + str(i)]), str(values[f'{model.names[item]}_NG_' + str(i)]), int(values[f'{model.names[item]}_Wn_' + str(i)]), int(values[f'{model.names[item]}_Wx_' + str(i)]), int(values[f'{model.names[item]}_Hn_' + str(i)]), int(values[f'{model.names[item]}_Hx_' + str(i)]),int(values[f'{model.names[item]}_PLC_' + str(i)]), int(values['OK_PLC_' + str(i)])))
            #conn.execute("INSERT INTO MYMODEL (ChooseModel,Camera, Weights,Confidence,Joined,Ok,Num,NG,WidthMin, WidthMax,HeightMin,HeightMax) \
            #    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (str(values['choose_model']),str(i)+ '_' +str(item) ,str(values['file_weights' + str(i)]), int(values['conf_thres' + str(i)]),str(values[f'{model.names[item]}_' + str(i)]), str(values[f'{model.names[item]}_OK_' + str(i)]), int(values[f'{model.names[item]}_Num_' + str(i)]), str(values[f'{model.names[item]}_NG_' + str(i)]), int(values[f'{model.names[item]}_Wn_' + str(i)]), int(values[f'{model.names[item]}_Wx_' + str(i)]), int(values[f'{model.names[item]}_Hn_' + str(i)]), int(values[f'{model.names[item]}_Hx_' + str(i)])))
        
    for row in cursor:
        if row[0] == choose_model:
            conn.execute("UPDATE MYMODEL SET have_1= ?,have_2= ?,have_3= ?,have_4= ?,have_5= ?,have_6= ?,have_7= ?,have_8= ?,have_9= ?,have_10= ?,folder_1= ?,folder_2= ?,folder_3= ?,folder_4= ?,folder_5= ?,folder_6= ?,folder_7= ?,folder_8= ?,folder_9= ?,folder_10 = ? WHERE ChooseModel = ? ",(str(values['have_save_1']),str(values['have_save_2']),str(values['have_save_3']),str(values['have_save_4']),str(values['have_save_5']),str(values['have_save_6']),str(values['have_save_7']),str(values['have_save_8']),str(values['have_save_9']),str(values['have_save_10']),str(values['save_1']),str(values['save_2']),str(values['save_3']),str(values['save_4']),str(values['save_5']),str(values['save_6']),str(values['save_7']),str(values['save_8']),str(values['save_9']),str(values['save_10']),choose_model))


    conn.commit()
    conn.close()


def make_window(theme):
    sg.theme(theme)

    file_weights = [('Weights (*.pt)', ('*.pt'))]

    right_click_menu = [[], ['Exit','Change Theme']]


    layout_main = [

        [
            sg.Input('',size=(50,1),font=('Helvetica',12), key='choose_folder_check',readonly= True, text_color='navy',enable_events= True),
            sg.FolderBrowse(size=(12,1), font=('Helvetica',10), key='folder_check',enable_events=True) ,
        ],

        [

        #1
        sg.Frame('',[
            [sg.Image(filename='', size=(image_width_display,image_height_display),key='image1',background_color='black'),

            sg.Frame('',[
                [sg.Button('Move 1', size=(10,1),  font=('Helvetica',14),disabled=False ,key= 'bt_save_1'),sg.Text(' '), sg.Button('Move 2', size=(10,1), font=('Helvetica',14), disabled=False, key= 'bt_save_2')],
                [sg.Input('di_vat',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),sg.Text(' '),sg.Input('me',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),],
                [sg.Text('')],
                [sg.Button('Move 3', size=(10,1), font=('Helvetica',14),disabled=False ,key= 'bt_save_3'), sg.Text(' '),sg.Button('Move 4', size=(10,1), font=('Helvetica',14),disabled=False,key= 'bt_save_4')],
                [sg.Input('nut',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),sg.Text(' '),sg.Input('nam_cham_cao',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),],
                [sg.Text('')],
                [sg.Button('Move 5', size=(10,1), font=('Helvetica',14),disabled=False ,key= 'bt_save_5'), sg.Text(' '),sg.Button('Move 6', size=(10,1), font=('Helvetica',14),disabled=False,key= 'bt_save_6')],
                [sg.Input('kim_cao',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),sg.Text(' '),sg.Input('di_vat_duoi',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),],
                [sg.Text('')],
                [sg.Button('Move 7', size=(10,1), font=('Helvetica',14),disabled=False ,key= 'bt_save_7'), sg.Text(' '),sg.Button('Move 8', size=(10,1), font=('Helvetica',14),disabled=False,key= 'bt_save_8')],
                [sg.Input('tray_bac_truc',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),sg.Text(' '),sg.Input('kim_bien_dang',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),],
                [sg.Text('')],
                [sg.Button('Move 9', size=(10,1), font=('Helvetica',14),disabled=False ,key= 'bt_save_9'), sg.Text(' '),sg.Button('Move 10', size=(10,1), font=('Helvetica',14),disabled=False,key= 'bt_save_10')],
                [sg.Input('nam_cham_cao',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),sg.Text(' '),sg.Input('kim_ri_set',size=(15,1),font=('Helvetica',15),text_color='navy',enable_events=True),],
                [sg.Text('')],
                [sg.Button('Back', size=(10,1), font=('Helvetica',14) ,key= 'back',enable_events=True), sg.Text(' '),sg.Button('Next', size=(10,1), font=('Helvetica',14),key= 'next')], 
                [sg.Text('')],
                [sg.Checkbox('Check',size=(2,1),font=('Helvetica',14), key='check_model1',enable_events=True,expand_x=True, expand_y=True),sg.Input('0',size=(8,1),font=('Helvetica',15),key= 'Num_index',text_color='navy',enable_events=True)],
                [sg.Text('')],
                [sg.Text(' '), sg.Combo(values=['1','2','3','4','5','6','7','8','9'], default_value='1',font=('Helvetica',20),size=(5, 100),text_color='navy',enable_events= True, key='choose_model')],
                [sg.Text('',font=('Helvetica',100), justification='center', key='result_cam1',expand_x=True)],
                [sg.Text('',font=('Helvetica',5), justification='center', key='time_cam1', expand_x=True)],
                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),
            ],
            [sg.Text('',justification='center' ,font= ('Helvetica',15),text_color='pink',expand_x=True,key='name_file')],           
        ]),
        ],
        # [
        # sg.Text('',justification='center' ,font= ('Helvetica',15),text_color='pink',expand_x=True,key='name_file'),

        # ],

    ] 




    layout_option1_1 = [
        [sg.Frame('',[
        [sg.Frame('',
        [   
            #[sg.Text('Location', size=(12,1), font=('Helvetica',15),text_color='red'), sg.Input(size=(60,1), font=('Helvetica',12), key='location_weights1',readonly= True, text_color='navy',enable_events= True),
            #sg.FolderBrowse(size=(15,1), font=('Helvetica',10),key= 'folder_browse1',enable_events=True)],
            [sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color='red'), sg.Input(size=(60,1), font=('Helvetica',12), key='file_weights1',readonly= True, text_color='navy',enable_events= True),
            #[sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color='red'), sg.Combo(values='', font=('Helvetica',12),size=(59, 30),text_color='navy',enable_events= True, key='file_weights1'),],
            sg.Frame('',[
                [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse1',enable_events=True, disabled=False)]
            ], relief= sg.RELIEF_FLAT),
            sg.Frame('',[
                [sg.Button('Change Model', size=(14,1), font=('Helvetica',10), disabled= True, key= 'Change_1')]
            ], relief= sg.RELIEF_FLAT),],
            [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='red'), sg.Slider(range=(1,100),orientation='h',size=(60,20),font=('Helvetica',11),disabled=False, key= 'conf_thres1'),]
        ], relief=sg.RELIEF_FLAT),
        ],
        [sg.Frame('',[
            [sg.Text('Name',size=(15,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Join',size=(7,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('OK',size=(7,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Num',size=(7,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('NG',size=(8,1),font=('Helvetica',15), text_color='red'),  
            sg.Text('Width Min',size=(11,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Width Max',size=(11,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Height Min',size=(11,1),font=('Helvetica',15), text_color='red'), 
            sg.Text('Height Max',size=(12,1),font=('Helvetica',15), text_color='red'),
            sg.Text('PLC',size=(11,1),font=('Helvetica',15), text_color='red'),
            sg.Text('Confidence',size=(11,1),font=('Helvetica',15), text_color='red')],
        ], relief=sg.RELIEF_FLAT)],
        [sg.Frame('',[
            [
                sg.Text(f'{model1.names[i1]}_1',size=(15,1),font=('Helvetica',15), text_color='yellow'), 
                sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key=f'{model1.names[i1]}_1',enable_events=True, disabled=False), 
                sg.Checkbox('',size=(5,5),font=('Helvetica',15),  key=f'{model1.names[i1]}_OK_1',enable_events=True, disabled=False), 
                sg.Input('1',size=(2,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Num_1',text_color='navy',enable_events=True, disabled=False), 
                sg.Text('',size=(4,1),font=('Helvetica',15), text_color='red'), 
                sg.Checkbox('',size=(5,5),font=('Helvetica',15),  key=f'{model1.names[i1]}_NG_1',enable_events=True, disabled=False), 
                sg.Input('0',size=(8,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Wn_1',text_color='navy',enable_events=True, disabled=False), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('100000',size=(8,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Wx_1',text_color='navy',enable_events=True, disabled=False), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('0',size=(8,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Hn_1',text_color='navy',enable_events=True, disabled=False), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('100000',size=(8,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Hx_1',text_color='navy',enable_events=True, disabled=False), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                sg.Input('0',size=(8,1),font=('Helvetica',15),key= f'{model1.names[i1]}_PLC_1',text_color='navy',enable_events=True, disabled=False), 
                sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                #sg.Input('0',size=(8,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Conf_1',text_color='navy',enable_events=True, disabled=False), 
                sg.Slider(range=(1,100),default_value=25,orientation='h',size=(30,20),font=('Helvetica',11), key= f'{model1.names[i1]}_Conf_1'),
            ] for i1 in range(len(model1.names))
        ], relief=sg.RELIEF_FLAT)],
        [sg.Text('  OK',size=(15,1),font=('Helvetica',15), text_color='yellow'),
        sg.Text(' '*230), 
        sg.Input('0',size=(8,1),font=('Helvetica',15),key= 'OK_PLC_1',text_color='navy',enable_events=True)],
        [sg.Text(' ')],
        [sg.Text(' '*250), sg.Button('Save Data', size=(12,1),  font=('Helvetica',12),key='SaveData1',enable_events=True)] 
        ])]
    ]
    
    
    layout_option1 = [[sg.Column(layout_option1_1, scrollable=True)]]

    layout_saveimg1 = [
        [sg.Frame('',[
                [sg.Text('Have save folder 1',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_1',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 1', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/1' ,font=('Helvetica',12), key='save_1',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_1',enable_events=True) ],
                [sg.Text('')],

                [sg.Text('Have save folder image 2',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_2',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 2', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/2' , font=('Helvetica',12), key='save_2',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_2',enable_events=True) ],
                [sg.Text('')],

                [sg.Text('Have save folder 3',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_3',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 3', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/3' ,font=('Helvetica',12), key='save_3',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_3',enable_events=True) ],
                [sg.Text('')],

                [sg.Text('Have save folder image 4',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_4',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 4', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/4' , font=('Helvetica',12), key='save_4',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_4',enable_events=True) ],
                [sg.Text('')],

                [sg.Text('Have save folder 5',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_5',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 5', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/5' ,font=('Helvetica',12), key='save_5',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_5',enable_events=True) ],
                [sg.Text('')],

        ], relief=sg.RELIEF_FLAT),
],
        ]

    layout_saveimg2 = [
        [sg.Frame('',[
                [sg.Text('Have save folder image 6',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_6',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 6', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/6' , font=('Helvetica',12), key='save_6',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_6',enable_events=True) ],
                [sg.Text('')],

                [sg.Text('Have save folder 7',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_7',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 7', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/7' ,font=('Helvetica',12), key='save_7',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_7',enable_events=True) ],
                [sg.Text('')],

                [sg.Text('Have save folder image 8',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_8',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 8', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/8' , font=('Helvetica',12), key='save_8',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_8',enable_events=True) ],
                [sg.Text('')],
    
                [sg.Text('Have save folder 9',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_9',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 9', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/9' ,font=('Helvetica',12), key='save_9',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_9',enable_events=True) ],
                [sg.Text('')],

                [sg.Text('Have save folder image 10',size=(35,1),font=('Helvetica',15), text_color='yellow'),sg.Checkbox('',size=(5,5),default=True,font=('Helvetica',15),  key='have_save_10',enable_events=True, disabled=False)], 
                [sg.T('Choose folder save image 10', font='Any 15', text_color = 'green')],
                [sg.Input(size=(50,1),default_text='C:/10' , font=('Helvetica',12), key='save_10',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(12,1), font=('Helvetica',10),key='save_folder_10',enable_events=True) ],
                [sg.Text('')],

        ], relief=sg.RELIEF_FLAT),
],
        ]




    layout_saveimg = [[sg.Column(layout_saveimg1, element_justification='c' ), sg.Column(layout_saveimg2, element_justification='c')]]

    layout_terminal = [[sg.Text("Anything printed will display here!")],
                      [sg.Multiline( font=('Helvetica',14), write_only=True, autoscroll=True, auto_refresh=True,reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True,expand_x=True,expand_y=True)]
                      ]
    
    layout = [[sg.TabGroup([[  sg.Tab('Main', layout_main),
                               sg.Tab('Option for model 1', layout_option1),
                    
                               sg.Tab('Save Image', layout_saveimg),
                               sg.Tab('Output', layout_terminal)]])
               ]]

    #layout[-1].append(sg.Sizegrip())
    window = sg.Window('HuynhLeVu', layout, location=(0,0),right_click_menu=right_click_menu,resizable=True).Finalize()
    #window.bind('<Configure>',"Configure")
    window.Maximize()

    return window



image_width_display = int(760*1.9)
image_height_display = int(480*1.9)

result_width_display = 500
result_height_display = 90


file_name_img = [("Img(*.jpg,*.png)",("*jpg","*.png"))]


recording1 = False

error_cam1 = True
a = 0


list_path = []
for path1 in glob.glob('C:/Check1/*.jpg'):
    list_path.append(path1)


# connected = False
# while connected == False:
#     connected = connect_plc('192.168.250.1')
#     print('connecting ....')
#     #event, values = window.read(timeout=20)

# print("connected plc")   


mypath1 = load_model(1)
model1 = torch.hub.load('./levu','custom', path= mypath1, source='local',force_reload =False)

move_al =0
index_i = 0
img1_test = os.path.join(os.getcwd(), 'img/imgtest.jpg')
result1 = model1(img1_test,416,0.25) 
print('model1 already')


al = True
choose_model = load_choosemodel()

themes = load_theme()
theme = themes[0]
window = make_window(theme)

window['choose_model'].update(value=choose_model)


try:
    load_all_sql(1,choose_model)
except:
    print(traceback.format_exc())
    window['time_cam1'].update(value= "Error data") 

index_path = 0
index_show = 1

connect_camera1 = False


connect_total = False

#removefile()



if connect_camera1 == True and connect_total == True:
    window['result_cam1'].update(value= 'Done', text_color='blue')


try:
    while True:
        event, values = window.read(timeout=20)

        for i1 in range(len(model1.names)):
            #if event == f'{model1.names[i1]}_1':
            if values[f'{model1.names[i1]}_1'] == False:
                window[f'{model1.names[i1]}_OK_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Num_1'].update(disabled=False)
                window[f'{model1.names[i1]}_NG_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Wn_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Wx_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Hn_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Hx_1'].update(disabled=False)
                window[f'{model1.names[i1]}_PLC_1'].update(disabled=False)

            elif values[f'{model1.names[i1]}_1'] == True:
                window[f'{model1.names[i1]}_OK_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Num_1'].update(disabled=False)
                window[f'{model1.names[i1]}_NG_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Wn_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Wx_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Hn_1'].update(disabled=False)
                window[f'{model1.names[i1]}_Hx_1'].update(disabled=False)
                window[f'{model1.names[i1]}_PLC_1'].update(disabled=False)

        for i1 in range(len(model1.names)):
            if event == f'{model1.names[i1]}_OK_1':
                if values[f'{model1.names[i1]}_OK_1'] == True:
                    window[f'{model1.names[i1]}_NG_1'].update(disabled=False)
                else:
                    window[f'{model1.names[i1]}_NG_1'].update(disabled=False)
            if event == f'{model1.names[i1]}_NG_1':
                if values[f'{model1.names[i1]}_NG_1'] == True:
                    window[f'{model1.names[i1]}_OK_1'].update(disabled=False)
                else:
                    window[f'{model1.names[i1]}_OK_1'].update(disabled=False)

        if keyboard.is_pressed('c'):
            image_width_display +=40
            image_height_display +=20

        if keyboard.is_pressed('t'):
            image_width_display -=40
            image_height_display -=20

        if event =='Exit' or event == sg.WINDOW_CLOSED :
            break



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
                if event_theme == sg.WIN_CLOSED:
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



        if event == 'file_browse1': 
            window['file_weights1'].update(value=values['file_browse1'])
            if values['file_browse1']:

                window['Change_1'].update(disabled=False)


        if event == 'choose_model':
            mychoose = values['choose_model']
            weight1 = ''
            conf_thres1 = 1


            have_1 = True
            have_2 = True
            have_3 = True
            have_4 = True
            have_5 = True
            have_6 = True
            have_7 = True
            have_8 = True
            have_9 = True
            have_10 = True


            Folder_1 = 'C:/1'
            Folder_2 = 'C:/2'
            Folder_3 = 'C:/3'
            Folder_4 = 'C:/4'  
            Folder_5 = 'C:/5'
            Folder_6 = 'C:/6'
            Folder_7 = 'C:/7'
            Folder_8 = 'C:/8'
            Folder_9 = 'C:/9'
            Folder_10 = 'C:/10'



            conn = sqlite3.connect('modeldb_1_Check.db')
            cursor = conn.execute("SELECT ChooseModel,Camera,Weights,Confidence,have_1,have_2,have_3,have_4,have_5,have_6,have_7,have_8,have_9,have_10,folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10,Joined,Ok,Num,NG,WidthMin,WidthMax,HeightMin,HeightMax from MYMODEL")
            for row in cursor:
                if row[0] == values['choose_model']:
 
                    mychoose = values['choose_model']
                    row1_a, row1_b = row[1].strip().split('_')
                    if row1_a == '1' and row1_b == '0':
                        weight1 = row[2]
                        conf_thres1 = row[3]

                        have_1 = str2bool(row[4])
                        have_2 = str2bool(row[5])
                        have_3 = str2bool(row[6])
                        have_4 = str2bool(row[7])
                        have_5 = str2bool(row[8])
                        have_6 = str2bool(row[9])
                        have_7 = str2bool(row[10])
                        have_8 = str2bool(row[11])
                        have_9 = str2bool(row[12])
                        have_10 = str2bool(row[13])
               
                        Folder_1 = row[14]
                        Folder_2 = row[15]
                        Folder_3 = row[16]
                        Folder_4 = row[17]  
                        Folder_5 = row[18]
                        Folder_6 = row[19]
                        Folder_7 = row[20]
                        Folder_8 = row[21]
                        Folder_9 = row[22]
                        Folder_10 = row[23]
                        
                     
                        model1 = torch.hub.load('./levu','custom', path= row[2], source='local',force_reload =False)

        
            window.close() 
            window = make_window(theme)

            window['file_weights1'].update(value=weight1)
            window['conf_thres1'].update(value=conf_thres1)
      
            window['choose_model'].update(value=mychoose)

            window['have_save_1'].update(value=have_1)    
            window['have_save_2'].update(value=have_2)
            window['have_save_3'].update(value=have_3)        
            window['have_save_4'].update(value=have_4)         
            window['have_save_5'].update(value=have_5)        
            window['have_save_6'].update(value=have_6)
            window['have_save_7'].update(value=have_7)        
            window['have_save_8'].update(value=have_8)        
            window['have_save_9'].update(value=have_9)        
            window['have_save_10'].update(value=have_10)

            window['save_1'].update(value=Folder_1)    
            window['save_2'].update(value=Folder_2)
            window['save_3'].update(value=Folder_3)        
            window['save_4'].update(value=Folder_4)         
            window['save_5'].update(value=Folder_5)        
            window['save_6'].update(value=Folder_6)
            window['save_7'].update(value=Folder_7)        
            window['save_8'].update(value=Folder_8)        
            window['save_9'].update(value=Folder_9)        
            window['save_10'].update(value=Folder_10)


            window['choose_model'].update(value=mychoose)


            cursor = conn.execute("SELECT ChooseModel,Camera,Weights,Confidence,have_1,have_2,have_3,have_4,have_5,have_6,have_7,have_8,have_9,have_10,folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10,Joined,Ok,Num,NG,WidthMin,WidthMax,HeightMin,HeightMax,PLC_NG,PLC_OK from MYMODEL")
            for row in cursor:
                if row[0] == values['choose_model']:
                    row1_a, row1_b = row[1].strip().split('_')
                    if row1_a == '1':
                        for item in range(len(model1.names)):
                            if int(row1_b) == item:
                                window[f'{model1.names[item]}_1'].update(value=str2bool(row[24]))
                                window[f'{model1.names[item]}_OK_1'].update(value=str2bool(row[25]))
                                window[f'{model1.names[item]}_Num_1'].update(value=str(row[26]))
                                window[f'{model1.names[item]}_NG_1'].update(value=str2bool(row[27]))
                                window[f'{model1.names[item]}_Wn_1'].update(value=str(row[28]))
                                window[f'{model1.names[item]}_Wx_1'].update(value=str(row[29]))
                                window[f'{model1.names[item]}_Hn_1'].update(value=str(row[30]))
                                window[f'{model1.names[item]}_Hx_1'].update(value=str(row[31]))
                                window[f'{model1.names[item]}_PLC_1'].update(value=str(row[32]))
                                window['OK_PLC_1'].update(value=str(row[33]))




            conn.close()

        if event == 'SaveData1':

            save_all_sql(model1,1,str(values['choose_model']))
            save_choosemodel(values['choose_model'])
            save_model(1,values['file_weights1'])
            sg.popup('Saved param model 1 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)



        if event == 'choose_folder_check':
            print('a')
            path_check = values['choose_folder_check'] + '/*.jpg'
            list_path = []
            for path1 in glob.glob(path_check):
                list_path.append(path1)

        if values['check_model1'] == True and al == True:
            try:

                if type(index_path) == int and index_path < len(list_path):
                    path1 = list_path[index_path]
                    for num,i in enumerate(list(reversed((path1)))):
                        if i == '\\':
                            index_i = -num
                            print(index_i)
                            break
                    name_image_current = path1[index_i:-4]
                    print('name : ',name_image_current)

                    img1_orgin = cv2.imread(path1)

                    img_save = img1_orgin
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
                                print('NG')
                                #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register_ng,b'\x00\x01',1)
                                cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                                window['result_cam1'].update(value= 'NG', text_color='red')
                                myresult1 = 1
                                

                        elif values[f'{model1.names[i1]}_NG_1'] == True:
                            if model1.names[i1] in names1:
                                print('NG')
                                #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register_ng,b'\x00\x01',1)
                                cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                                window['result_cam1'].update(value= 'NG', text_color='red')    
                                myresult1 = 1         
                                    

                    if myresult1 == 0:
                        print('OK')
                        check_ok = 1
                        #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,(3000).to_bytes(2, byteorder='big') + b'\x00',b'\x00\x01',1)
                        cv2.putText(show1, 'OK',(result_width_display+100,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam1'].update(value= 'OK', text_color='green')

                    cv2.putText(show1, str(index_show),(50,60),cv2.FONT_HERSHEY_COMPLEX, 2,(255,0,0),2)
                    imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                    window['image1'].update(data= imgbytes1)
                    window['name_file'].update(value= str(name_image_current))
                    al = False
                    next = 2
            except:
                #sg.popup('Anh da xoa')
                if next == 1:
                    if index_path < len(list_path):
                        index_path += 1
                        index_show += 1
                    else:
                        sg.popup('Đã hết ảnh')
                else:
                    if index_path >= 0:
                        index_path -= 1
                        index_show -= 1
                    else:
                        sg.popup('Ảnh đầu tiên')       
                


        if values['have_save_1']:
            if event == 'bt_save_1':
                cv2.imwrite(values['save_1']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')

        if values['have_save_2']:
            if event == 'bt_save_2':
                cv2.imwrite(values['save_2']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')

        if values['have_save_3']:
            if event == 'bt_save_3':
                cv2.imwrite(values['save_3']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')

        if values['have_save_4']:
            if event == 'bt_save_4':
                cv2.imwrite(values['save_4']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')

        if values['have_save_5']:
            if event == 'bt_save_5':
                cv2.imwrite(values['save_5']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')

        if values['have_save_6']:
            if event == 'bt_save_6':
                cv2.imwrite(values['save_6']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')


        if values['have_save_7']:
            if event == 'bt_save_7':
                cv2.imwrite(values['save_7']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')

        if values['have_save_8']:
            if event == 'bt_save_8':
                cv2.imwrite(values['save_8']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')

        if values['have_save_9']:
            if event == 'bt_save_9':
                cv2.imwrite(values['save_9']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')

        if values['have_save_10']:
            if event == 'bt_save_10':
                cv2.imwrite(values['save_10']  + '/' + name_image_current + '.jpg',img_save)
                move_al = 1
                #os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')

        if event == 'Num_index':
            if move_al:
                os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')
                move_al = 0
            al = True
            if values['Num_index'] != '' and values['Num_index'].isdigit() :
                index_path = int(values['Num_index']) - 1
                index_show = int(values['Num_index']) 

        if event == 'next' or keyboard.is_pressed('d'):
            if move_al:
                os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')
                move_al = 0
            al = True
            next = 1
            if index_path < len(list_path):
                index_path += 1
                index_show += 1
            else:
                #pass
                sg.popup('Đã hết ảnh')
        if event == 'back' or keyboard.is_pressed('a'):
            if move_al:
                os.remove(values['choose_folder_check'] + '/' + name_image_current + '.jpg')
                move_al = 0
            next = 0
            al = True
            if index_path >= 0:
                index_path -= 1
                index_show -= 1
            else:
                #pass
                sg.popup('Ảnh đầu tiên')


        elif event == 'Stop1':
            recording1 = False 
            imgbytes1 = np.zeros([100,100,3],dtype=np.uint8)
            imgbytes1 = cv2.resize(imgbytes1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
            imgbytes1 = cv2.imencode('.png',imgbytes1)[1].tobytes()
            window['image1'].update(data=imgbytes1)
            window['result_cam1'].update(value='')



        if event == 'Pic1':
            dir_img1 = sg.popup_get_file('Choose your image 1',file_types=file_name_img,keep_on_top= True)
            if dir_img1 not in ('',None):
                pic1 = Image.open(dir_img1)
                img1_resize = pic1.resize((image_width_display,image_height_display))
                imgbytes1 = ImageTk.PhotoImage(img1_resize)
                window['image1'].update(data= imgbytes1)
                window['Detect1'].update(disabled= False)         

        if event == 'Change_1':

            list_variable = [[0]*11 for i in range(len(model1.names))]

            for i,item in enumerate(range(len(model1.names))):
                list_variable[i][0] = model1.names[i]

                list_variable[i][1] = values[f'{model1.names[item]}_1']
                list_variable[i][2] = values[f'{model1.names[item]}_OK_1'] 
                list_variable[i][3] = values[f'{model1.names[item]}_Num_1'] 
                list_variable[i][4] = values[f'{model1.names[item]}_NG_1'] 
                list_variable[i][5] = values[f'{model1.names[item]}_Wn_1'] 
                list_variable[i][6] = values[f'{model1.names[item]}_Wx_1'] 
                list_variable[i][7] = values[f'{model1.names[item]}_Hn_1'] 
                list_variable[i][8] = values[f'{model1.names[item]}_Hx_1'] 
                list_variable[i][9] = values[f'{model1.names[item]}_PLC_1'] 
                list_variable[i][10] = values['OK_PLC_1']
    


            mypath1 = values['file_weights1']
            model1= torch.hub.load('./levu','custom',path=mypath1,source='local',force_reload=False)

            mychoose = values['choose_model']
            weight1 = values['file_weights1']
            conf_thres1 = values['conf_thres1'] 

            have_1 = values['have_save_1']
            have_2 = values['have_save_2']
            have_3 = values['have_save_3']
            have_4 = values['have_save_4']
            have_5 = values['have_save_5']
            have_6 = values['have_save_6']
            have_7 = values['have_save_7']
            have_8 = values['have_save_8']
            have_9 = values['have_save_9']
            have_10 = values['have_save_10']

            Folder_1 = values['save_1']
            Folder_2 = values['save_2']
            Folder_3 = values['save_3']
            Folder_4 = values['save_4']  
            Folder_5 = values['save_5']
            Folder_6 = values['save_6']
            Folder_7 = values['save_7']
            Folder_8 = values['save_8']
            Folder_9 = values['save_9']
            Folder_10 = values['save_10']


            window.close() 
            window = make_window(theme)

            window['choose_model'].update(value=mychoose)
            window['file_weights1'].update(value=weight1)
            window['conf_thres1'].update(value=conf_thres1)
            
            window['have_save_1'].update(value=have_1)    
            window['have_save_2'].update(value=have_2)
            window['have_save_3'].update(value=have_3)        
            window['have_save_4'].update(value=have_4)         
            window['have_save_5'].update(value=have_5)        
            window['have_save_6'].update(value=have_6)
            window['have_save_7'].update(value=have_7)        
            window['have_save_8'].update(value=have_8)        
            window['have_save_9'].update(value=have_9)        
            window['have_save_10'].update(value=have_10)

            window['save_1'].update(value=Folder_1)    
            window['save_2'].update(value=Folder_2)
            window['save_3'].update(value=Folder_3)        
            window['save_4'].update(value=Folder_4)         
            window['save_5'].update(value=Folder_5)        
            window['save_6'].update(value=Folder_6)
            window['save_7'].update(value=Folder_7)        
            window['save_8'].update(value=Folder_8)        
            window['save_9'].update(value=Folder_9)        
            window['save_10'].update(value=Folder_10)


            window['choose_model'].update(value=mychoose)



            for i, item in enumerate(range(len(model1.names))):
                for name_label in model1.names:
                    if len(model1.names) <= len(list_variable):
                        if name_label == list_variable[i][0]:

                            window[f'{model1.names[item]}_1'].update(value= list_variable[i][1])
                            window[f'{model1.names[item]}_OK_1'].update(value= list_variable[i][2])
                            window[f'{model1.names[item]}_Num_1'].update(value= list_variable[i][3])
                            window[f'{model1.names[item]}_NG_1'].update(value= list_variable[i][4])
                            window[f'{model1.names[item]}_Wn_1'].update(value= list_variable[i][5])
                            window[f'{model1.names[item]}_Wx_1'].update(value= list_variable[i][6])
                            window[f'{model1.names[item]}_Hn_1'].update(value= list_variable[i][7])
                            window[f'{model1.names[item]}_Hx_1'].update(value= list_variable[i][8])
                            window[f'{model1.names[item]}_PLC_1'].update(value= list_variable[i][9])
                            window['OK_PLC_1'].update(value= list_variable[i][10])





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
                    conf1 = table1['confidence'][item] *100
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
                                elif conf1  < int(values[f'{model1.names[i1]}_Conf_1']):
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
                    if values[f'{model1.names[i1]}_1'] == True:
                        if values[f'{model1.names[i1]}_OK_1'] == True:
                            len_name1 = 0
                            for name1 in names1:
                                if name1 == model1.names[i1]:
                                    len_name1 +=1
                            if len_name1 != int(values[f'{model1.names[i1]}_Num_1']):
                                print('NG')
                                cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                                window['result_cam1'].update(value= 'NG', text_color='red')
                                myresult1 = 1
                                break

                        if values[f'{model1.names[i1]}_NG_1'] == True:
                            if model1.names[i1] in names1:
                                print('NG')
                                cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
                                window['result_cam1'].update(value= 'NG', text_color='red')    
                                myresult1 = 1         
                                break    

                if myresult1 == 0:
                    print('OK')
                    cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
                    window['result_cam1'].update(value= 'OK', text_color='green')

                imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                window['image1'].update(data= imgbytes1)

            
            except:
                print(traceback.format_exc())
                sg.popup_annoying("Don't have image or parameter wrong", font=('Helvetica',14),text_color='red')
            
            t2 = time.time() - t1
            print(t2)
            time_cam1 = str(int(t2*1000)) + 'ms'
            window['time_cam1'].update(value= time_cam1, text_color='black') 
            print('---------------------------------------------') 



    window.close() 

except Exception as e:
    print(traceback.print_exc())
    str_error = str(e)
    sg.popup(str_error,font=('Helvetica',15), text_color='red',keep_on_top= True)
              
