import time
from win10toast import ToastNotifier

file_path = r'D:\CHECK_DATETIME_CHAU_CHOI_ERROR.txt'

previous_content = ""

def check_and_notify(file_path, previous_content):

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    if content != previous_content:
        toaster = ToastNotifier()
        toaster.show_toast("THÔNG BÁO KHẨN CẤP !!!!!!", "CÓ 1 CON HÀNG VỚI HẠNG MỤC NG* NGHI VẤN LẪN LỘN CHẤU CHỔI, HÃY XUỐNG CHECK FILE!", duration=10)
        return content 

    return previous_content

if __name__ == "__main__":
    while True:
        previous_content = check_and_notify(file_path, previous_content)
        time.sleep(1)  
