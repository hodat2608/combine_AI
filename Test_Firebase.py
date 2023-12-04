import sqlite3
import firebase_admin
from firebase_admin import credentials, db
import base64, cv2
from datetime import datetime, timedelta

def get_shift():
    now = datetime.now()
    if 7 <= now.hour < 19:
        shift = 'D'
    elif 0 <= now.hour < 7:
        # Ca đêm kết thúc vào lúc 7 giờ sáng ngày hôm nay
        shift = 'N'
        now -= timedelta(days=1)
    else:
        shift = 'N'
    return now.strftime('%Y-%m-%d-' + shift)

def mahoa(ddan,ten):
    image = cv2.imread(ddan + '/' + ten + '.jpg')
    # Get the dimensions of the image
    height, width = image.shape[:2]
    # Calculate the new dimensions of the image
    new_height = int(height * 0.25)
    new_width = int(width * 0.25)
    # Resize the image
    img = cv2.resize(image, (new_width, new_height))
    
    retval, buffer = cv2.imencode('.jpg', img)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# Initialize Firebase
cred = credentials.Certificate('firebase-adminsdk.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://soy-pillar-184015.firebaseio.com/'
})

# Connect to SQLite database
conn = sqlite3.connect('//ccai0004/combine/defect.db')
cursor = conn.cursor()
anh = ''
DE = 'DEN'
while True:
    # Retrieve data from SQLite database
    cursor.execute('SELECT * FROM PHEPHAM ORDER BY ANH DESC')
    row = cursor.fetchone()
    try:
        if row[1] != anh:
            opt = ''
            if 'C1' in row[0]:
                opt = 'OPT1'
            elif 'CD' in row[0]:
                opt = 'OPT2'
            else:
                opt = 'OPT3'
            img = mahoa('X:/' + DE + '/NG/' + opt,row[0])
                
            # Push data to Firebase
            ref = db.reference('X75/' + get_shift())
            ref.push({
                'name': row[0],
                'faults': row[1],
                'image': img
                })
            print(opt,row[1])
            anh = row[1]
        else:
            print('Runing.....................',end='\r')
    except:
        if DE == 'DEN':
            DE = 'TRANG'
        else:
            DE == 'DEN'
        pass
# Close SQLite connection
conn.close()
