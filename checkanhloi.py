from PIL import Image
from glob import glob
import os
from TOAN import Test


source_folder_copy_folder = Test.getPath1()
def check_image(source_folder_copy_folder):
    c = 0
    image = glob(source_folder_copy_folder+'*.jpg')
    for i in image:
        try:
            image = Image.open(i)
            image.verify()
            c+=1
            continue
        except (IOError, SyntaxError) as e:
            print(i)
            # os.remove(i)
            c+=1
            print(c,i)
    print('complete')




check_image(source_folder_copy_folder)
