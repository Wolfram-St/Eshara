import cv2 as cv
import numpy as np
import pickle as pkl

def build_squares(img):
    x,y,w,h = 420,140,10,10
    d = 10
    img_crop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if img_crop is None:
                img_crop = img[y:y+h, x:x+w]
            else:
                img_crop = np.hstack((img_crop, img[y:y+h, x:x+w]))
            cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            x+=w+d
        if crop is None:
            crop = img_crop
        else:
            crop = np.vstack((crop, img_crop)) 
        img_crop = None
        x = 420
        y+=h+d
    return crop

def get_hand_hist():
    cam = cv.VideoCapture(1)
    if cam.read()[0]==False:
        cam = cv.VideoCapture(0)
    x,y,w,h = 300,100,300,300
    flag_pressed_c, flag_pressed_s = False, False
    img_crop = None
    hist = None
    calibration_samples = []
    
    while True:
        img = cam.read()[1]
        img = cv.flip(img, 1)
        img = cv.resize(img, (640, 480))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)  # Better color space
        
        keypress = cv.waitKey(1)
        if keypress == ord('c'):        
            if img_crop is not None:
                # Add dynamic sample weighting
                calibration_samples.append(img_crop)
                print(f"Collected {len(calibration_samples)} samples")
                
                # Merge samples and create histogram
                merged_samples = np.vstack(calibration_samples)
                hsv_crop = cv.cvtColor(merged_samples, cv.COLOR_BGR2YCrCb)
                new_hist = cv.calcHist([hsv_crop], [1,2], None, [256,256], [0,256,0,256])
                
                # Blend with previous histogram if exists
                if hist is not None:
                    hist = cv.addWeighted(hist, 0.7, new_hist, 0.3, 0)
                else:
                    hist = new_hist
                    
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                flag_pressed_c = True
                
        elif keypress == ord('s'):
            if hist is not None and len(calibration_samples) >= 5:
                flag_pressed_s = True  
                break
            else:
                print("Need at least 5 samples! Press 'c' more times")
                
        if flag_pressed_c:    
            # Show real-time threshold feedback
            dst = cv.calcBackProject([hsv], [1,2], hist, [0,256,0,256], 1)
            dst1 = dst.copy()
            disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))
            cv.filter2D(dst,-1,disc,dst)
            blur = cv.GaussianBlur(dst, (11,11), 0)
            blur = cv.medianBlur(blur, 15)
            ret,thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            thresh = cv.merge((thresh,thresh,thresh))
            cv.imshow("Thresh Feedback", thresh)
            
        if not flag_pressed_s:
            img_crop = build_squares(img)
            
        cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
        cv.imshow("Hand Histogram", img)
        
    cam.release()
    cv.destroyAllWindows()
    with open("hist", "wb") as f:
        pkl.dump(hist, f)
    print("Calibration saved with", len(calibration_samples), "environment samples")

get_hand_hist()
