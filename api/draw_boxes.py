import os
from tempfile import NamedTemporaryFile

from PIL import Image, ImageDraw

def draw_boxes(bboxes):
    preds = []
    for boxes in bboxes:
        img = boxes["image"]
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size
        n_full = 0
        for box in boxes["bboxes"]:
            cls = box["class"]
            n_full += cls
            center_xy = box["center"]
            size = box["size"]
            lx = (center_xy[0]-size[0]/2)*img_w - 10
            rx = (center_xy[0]+size[0]/2)*img_w + 10
            ly = (center_xy[1]-size[1]/2)*img_h - 10
            ry = (center_xy[1]+size[1]/2)*img_h + 10
            
            color = "blue"
            print(color)
            draw.rectangle([lx, ly, rx, ry], outline=color, width=3)
        
        # with NamedTemporaryFile(mode="w", newline='', encoding="utf-8", suffix=".png") as f:
        img.save("tmp.png")
        n_all = len(boxes["bboxes"])
        # print(n_full/n_all > 0.5 if n_all else False, n_full, n_all)
        preds.append({"overall":n_full/n_all > 0.5 if n_all else False,
            "n_full":n_full,
            "n_all":n_all})
    return preds