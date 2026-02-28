from PIL import Image

# 假设 x 是一个 256x256 的图像或张量

# 定义裁剪的具体区域坐标
eye1_top = 94
eye1_bottom = 94 + 64
eye1_left = 54
eye1_right = 54 + 64

eye2_top = 94
eye2_bottom = 94 + 64
eye2_left = 128
eye2_right = 128 + 64

nose_top = 100
nose_bottom = 100 + 96
nose_left = 86
nose_right = 86 + 96

mouth_top = 151
mouth_bottom = 151 + 96
mouth_left = 85
mouth_right = 85 + 96

# 打开图像（假设 x 是一个 PIL Image 对象）
image = Image.open('/home/nwu-kiki/mydisk/pycharmprojects/BBDM+Trasformer+light/BBDM-main/LG-BBDM-2/train/B/66.png')  # 替换成你的图像文件路径
print(1)
# 裁剪图像的指定区域
eye1_x = image.crop((eye1_left, eye1_top, eye1_right, eye1_bottom))
eye2_x = image.crop((eye2_left, eye2_top, eye2_right, eye2_bottom))
nose_x = image.crop((nose_left, nose_top, nose_right, nose_bottom))
mouth_x = image.crop((mouth_left, mouth_top, mouth_right, mouth_bottom))

# 可选：显示裁剪后的图像（用于调试或可视化）
eye1_x.show()
eye2_x.show()
nose_x.show()
mouth_x.show()

# 可选：保存裁剪后的图像（用于后续处理或存档）
eye1_x.save('eye1_crop.jpg')  # 替换成你想要保存的路径和文件名
eye2_x.save('eye2_crop.jpg')
nose_x.save('nose_crop.jpg')
mouth_x.save('mouth_crop.jpg')
