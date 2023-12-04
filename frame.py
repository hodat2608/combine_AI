import PySimpleGUI as sg

# Tạo layout cho 2 cột
column1_layout = [
    [sg.Text(' ', font=('Helvetica', 5))],
    [sg.Image(filename='', size=(200, 200), key='image1', background_color='black')],
    [sg.Image(filename='', size=(200, 200), key='image2', background_color='black')]
]

column2_layout = [
    [sg.Text(' ', font=('Helvetica', 5))],
    [sg.Image(filename='', size=(200, 200), key='image3', background_color='black')],
    [sg.Image(filename='', size=(200, 200), key='image4', background_color='black')]
]

# Tạo layout cho Frame chứa 2 cột
frame_layout = [
    [sg.Frame('', column1_layout,)],
    [sg.Frame('', column2_layout,)]
]

# Tạo giao diện chính
layout = [
    [sg.Frame('', frame_layout, title='4 sg.Image')],
    # ...Thêm phần layout khác nếu cần
]

window = sg.Window('Sắp xếp 4 sg.Image', layout, finalize=True)

# Main event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

# Đóng cửa sổ
window.close()
