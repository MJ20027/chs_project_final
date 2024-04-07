import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def calculate_metrics(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_true = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces_true) > 0 and faces_true[0][0] < image.shape[1] // 2 and faces_true[0][1] < image.shape[0] // 2:
        detected = True
    else:
        detected = False
    
    if detected:
        true_positive = 1
        false_positive = 0
        false_negative = 0
    else:
        true_positive = 0
        false_positive = 1
        false_negative = 1 if len(faces_true) > 0 else 0
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    return true_positive, false_positive, false_negative, precision, recall

images_directory = 'Images_analysis'

tp_total = 0
fp_total = 0
fn_total = 0
precision_total = 0
recall_total = 0

for filename in os.listdir(images_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'): 
        image_path = os.path.join(images_directory, filename)
        tp, fp, fn, precision, recall = calculate_metrics(image_path)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        precision_total += precision
        recall_total += recall

total_images = len(os.listdir(images_directory))
average_precision = precision_total / total_images
average_recall = recall_total / total_images

print(f'Total images: {total_images}')
print(f'True Positives: {tp_total}')
print(f'False Positives: {fp_total}')
print(f'False Negatives: {fn_total}')
print(f'Average Precision: {average_precision:.2f}')
print(f'Average Recall: {average_recall:.2f}')