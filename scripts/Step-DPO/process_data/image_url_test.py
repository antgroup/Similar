import os, base64
import requests as req
from PIL import Image
from io import BytesIO

response = req.get("https://pic.imgdb.cn/item/6711306dd29ded1a8c6da9ac.png")

# 内存中打开图片
image = Image.open(BytesIO(response.content))

print("image = ", image)

# 图片的base64编码
ls_f = base64.b64encode(BytesIO(response.content).read())

# base64编码解码
imgdata = base64.b64decode(ls_f)

# 图片文件保存
file = open('test.jpg','wb')
file.write(imgdata)
file.close()