from PIL import Image, ImageDraw

img_path = "D:\\RynnBrain\\data\\pull.jpg"  # 你的第3帧图片
affordance =   (111, 741)  # 模型输出
bbox = ((698, 308), (805, 377))  # 模型输出

img = Image.open(img_path).convert("RGB")
w, h = img.size

def denorm(pt):
    x, y = pt
    return round(x / 1000 * w), round(y / 1000 * h)

# 反归一化
ax, ay = denorm(affordance)
(x1, y1), (x2, y2) = denorm(bbox[0]), denorm(bbox[1])

draw = ImageDraw.Draw(img)

# 画框
draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

# 画点
r = 6
draw.ellipse([ax-r, ay-r, ax+r, ay+r], fill="red", outline="red")

out = "D:\\RynnBrain\\data\\output.jpg"
img.save(out)
print("saved:", out)