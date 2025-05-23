import cv2
import numpy as np
import pickle, os, sqlite3, random

image_x, image_y = 50, 50

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def init_create_folder_database():
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        conn.execute("CREATE TABLE gesture (g_id INTEGER PRIMARY KEY, g_name TEXT NOT NULL)")
        conn.commit()

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    try:
        conn.execute("INSERT INTO gesture VALUES (?,?)", (g_id, g_name))
    except sqlite3.IntegrityError:
        if input("g_id exists. Update? (y/n): ").lower() == 'y':
            conn.execute("UPDATE gesture SET g_name=? WHERE g_id=?", (g_name, g_id))
    conn.commit()

def store_images(g_id):
    hist = get_hand_hist()
    
    # Improved camera initialization
    cam = None
    for idx in [0, 1, 2]:
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            break
    if not cam or not cam.isOpened():
        print("Camera Error!")
        return

    x, y, w, h = 300, 100, 300, 300
    create_folder(f"gestures/{g_id}")
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    
    while True:
        ret, img = cam.read()
        if not ret: break
        
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Better color space
        
        # Improved backprojection
        mask = cv2.calcBackProject([hsv], [1,2], hist, [0,256,0,256], 1)
        mask = cv2.medianBlur(mask, 15)
        
        # Enhanced thresholding
        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        
        roi = thresh[y:y+h, x:x+w]
        
        # Contour handling with error checking
        contours = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(main_contour)
                hand_roi = roi[y1:y1+h1, x1:x1+w1]
                
                # Better image padding
                pad_h = max(0, (w1 - h1) // 2)
                pad_v = max(0, (h1 - w1) // 2)
                hand_roi = cv2.copyMakeBorder(hand_roi, pad_h, pad_h, pad_v, pad_v, 
                                            cv2.BORDER_CONSTANT, value=0)
                
                hand_roi = cv2.resize(hand_roi, (image_x, image_y))
                if random.random() > 0.5:
                    hand_roi = cv2.flip(hand_roi, 1)
                
                cv2.imwrite(f"gestures/{g_id}/{pic_no}.jpg", hand_roi)
                pic_no += 1
                cv2.putText(img, "Capturing...", (30, 60), 
                           cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))

        # Visualization
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, f"Captured: {pic_no}/1200", (30, 400), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1, (127, 127, 255))
        
        cv2.imshow("Live Feed", img)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("ROI", roi)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            flag_start_capturing = not flag_start_capturing
            frames = 0
        elif key == 27 or pic_no >= 1200:
            break
            
        frames += 1 if flag_start_capturing else 0

    cam.release()
    cv2.destroyAllWindows()

# Initialize and run
init_create_folder_database()
g_id = int(input("Gesture ID: "))
g_name = input("Gesture Name: ")
store_in_db(g_id, g_name)
store_images(g_id)
