from glob import glob
import os, cv2, torch, time, datetime, shutil
import numpy as np 
import pandas as pd
import PySimpleGUI as sg
layout_cam1 = [
       [
        sg.Frame('PATH',[
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

window = sg.Window('HuynhLeVu', layout_cam1, location=(0,0),resizable=True).Finalize()
    
while True:
    event, values = window.read(timeout=20)

    