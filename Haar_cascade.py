import cv2

img_path = 'image.jpg'
img = cv2.imread(img_path)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

haar_values = []
for (x, y, w, h) in faces:
    roi = gray_img[y:y+h, x:x+w]  
    dark_pixels_sum = cv2.sumElems(roi)[0]
    dark_pixels_count = cv2.countNonZero(roi)
    dark_pixels_avg = dark_pixels_sum / dark_pixels_count
    light_pixels_sum = cv2.sumElems(cv2.bitwise_not(roi))[0]
    light_pixels_count = roi.shape[0] * roi.shape[1] - dark_pixels_count
    light_pixels_avg = light_pixels_sum / light_pixels_count
    haar_value = dark_pixels_avg - light_pixels_avg
    haar_values.append(haar_value)

print("Haar values:", haar_values)