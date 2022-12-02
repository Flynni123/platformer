import numpy as np


speed = 1
size = (1920, 1080)  # 1920, 1080
unscaledSize = (192, 108)  # 192, 108

assert size[0]/unscaledSize[0] == size[1]/unscaledSize[1]
scaleFactor = size[0]/unscaledSize[0]

dtype = np.float32

debug = False

character = False
physics = True
light = True
