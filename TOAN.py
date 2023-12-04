import PySimpleGUI as sg

class Test:
    def getPath1():
        TM = ''
        while TM=='':
            TM = sg.PopupGetFolder(message='Hãy chọn thư mục',title='path1_file_goc')
            if TM != '' and TM != None:
                TM = TM.replace('\\','/') + '/'
            elif TM =='':
                sg.popup_error('Bạn chưa nhập folder')
            else:
                sg.popup('Bạn đã Cancel')
                quit()
        return TM
    def getPath2():
        TM = ''
        while TM=='':
            TM = sg.PopupGetFolder(message='Hãy chọn thư mục',title='path1_file_goc')
            if TM != '' and TM != None:
                TM = TM.replace('\\','/') + '/'
            elif TM =='':
                sg.popup_error('Bạn chưa nhập folder')
            else:
                sg.popup('Bạn đã Cancel')
                quit()
        return TM
    def getPath3():
        TM = ''
        while TM=='':
            TM = sg.PopupGetFolder(message='Hãy chọn thư mục',title='path3_folder save')
            if TM != '' and TM != None:
                TM = TM.replace('\\','/') + '/'
            elif TM =='':
                sg.popup_error('Bạn chưa nhập folder')
            else:
                sg.popup('Bạn đã Cancel')
                quit()
        return TM
    
