import cv2
from roboflow import Roboflow
import supervision as sv
import tkinter as tk
from PIL import Image, ImageTk

# Roboflow API anahtarınızı buraya yerleştirin
rf = Roboflow(api_key="al0ZdOcQMp8RkZTJ92s3")

# Roboflow'dan kullanılacak proje ve modeli seçin
project = rf.workspace().project("earthquake-damage-detection-xmfgr")
model = project.version(1).model

# Tkinter penceresi oluşturun
root = tk.Tk()
root.title("Object Detection")

# Etiketleyici ve kutu işaretleyiciyi oluşturun
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

# Görüntüyü göstermek için bir Canvas oluşturun
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

def detect_objects(image):
    # Modelleme yapmak için görüntüyü Roboflow'a gönderin ve sonuçları alın
    result = model.predict(image, confidence=40, overlap=30).json()

    # Sonuçlardan etiketleri ve tespitleri alın
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_roboflow(result)

    # Görüntü üzerinde tespitleri işaretleyin
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image

def process_image(image_path):
    # Fotoğrafı okuyun
    image = cv2.imread(image_path)

    # Görüntüyü işleyin ve nesneleri tespit edin
    annotated_image = detect_objects(image)

    # OpenCV görüntüsünü PIL formatına dönüştürün
    img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    img = ImageTk.PhotoImage(image=img)

    # Canvas'a görüntüyü gösterin
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.img = img

# Doğrudan fotoğrafın yolunu belirtin
image_path = "../depremrobo2/yeni-haber-basligi_0e71b268.jpg"
process_image(image_path)

# Ana döngüyü başlatın
root.mainloop()
