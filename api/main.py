import datetime
import threading
import matplotlib.pyplot as plt
import numpy as np
import requests
import cv2
from io import BytesIO
from os.path import join
from PIL import Image
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List
import draw_boxes
from detect_box import run

cameras = requests.get("http://192.168.125.30:8000/cameras").json()
areas = cameras[0]["areas"]

def distance(line, point):
    x1, y1, x2, y2 = line
    x3, y3 = point
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return abs(a * x3 - y3 + b) / (a ** 2 + 1) ** 0.5

def distance2(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def is_in_polygon(point, polygon):
    x, y = point
    for i in range(len(polygon) - 1):
        if (polygon[i][0] <= x <= polygon[i + 1][0] or polygon[i][0] >= x >= polygon[i + 1][0]) and (polygon[i][1] <= y <= polygon[i + 1][1] or polygon[i][1] >= y >= polygon[i + 1][1]):
            return True
    return False


res_area = {"time":0,"areas":[],"camera":[{"uuid":cameras[0]["uuid"]}]}
res = {}
res_cam = []

arr = []
for a in areas:
    res[a["uuid"]] = 0
    for p in a["points"]:
        arr.append([p["x"],p["y"]])

penta = np.array([arr], np.int32)

def grab():
    # res_area
    global res_area,res,res_cam
    arrrr = np.zeros([1920, 1280], dtype=int)
    res_area = {"areas":res,"camera":[{"uuid":cameras[0]["uuid"],"points":res_cam}]}
    # requests.post("http://192.168.125.30:8000/stats", json=res_area)

    for i in res_cam:
        arrrr[i["x"]][i["y"]] = 50
    plt.imshow(arrrr, cmap='hot', interpolation='nearest')
    plt.savefig('demo.png')

    print(res_area)
    res_area = {"time": 0, "areas": [], "camera": [{"uuid": cameras[0]["uuid"], "points":[]}]}
    res = {}
    for a in areas:
        res[a["uuid"]] = 0
    res_cam = []

class MyThread(threading.Thread):
    def __init__(self, event):
        threading.Thread.__init__(self)
        self.stopped = event

    def run(self):
        while not self.stopped.wait(3):
            grab()
            # call a function

stopFlag = threading.Event()
thread = MyThread(stopFlag)
thread.start()

while True:
    resp = requests.get("http://169.254.54.9/oneshotimage1", stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite("imgdd.png", frame)
    bboxes = run(source="imgdd.png")

    arr = []
    for a in areas:
        arr = []
        for p in a["points"]:
            arr.append([p["x"], p["y"]])
        penta = np.array([arr], np.int32)
        cv2.polylines(frame, [penta], True, (255, 255, 0), thickness=3)

    for *xyxy, conf, cls in reversed(bboxes["det"]):
        dot = (int((int(xyxy[0].numpy())+int(xyxy[2].numpy()))/2),int((int(xyxy[1].numpy())+int(xyxy[3].numpy()))/2))
        res_cam.append({"x":dot[0],"y":int(xyxy[3].numpy())})
        c = int(cls)
        label = f'{bboxes["names"][c]} {conf:.2f}'
        # print(label)
        if bboxes["names"][c] == "person":
            cv2.rectangle(frame, (int(xyxy[0].numpy()), int(xyxy[1].numpy())), (int(xyxy[2].numpy()), int(xyxy[3].numpy())), (0, 255, 0), 3)
            cv2.circle(frame, dot, 3, (0, 0, 255),7)

        min = [99999999999, 99999999999]
        area_min = {"name":"","distance":999999}
        for a in areas:
            min_local = [0, 0]
            buf = []

            if is_in_polygon(dot,arr):
                res[a["uuid"]] += 1
                break


            for i, p in enumerate(a["points"]):
                if i==0:
                    buf = [p["x"], p["y"]]
                    continue

                arr.append([p["x"], p["y"]])
                min_local[0] = distance2([p["x"], p["y"]],dot) + distance2(buf,dot)
                min_local[1] = distance([p["x"], p["y"], buf[0], buf[1]], dot)
                buf = [p["x"], p["y"]]

            if min_local[1] <= min[1]:
                area_min["name"] = a["uuid"]
                area_min["distance"] = min_local[1]
                min = min_local

        if area_min["distance"] <= 10:
            res[area_min["name"]] += 1


    # print(res)




    cv2.imshow("Output Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()




