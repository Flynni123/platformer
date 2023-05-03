import numpy as np

speed = 1
runFac = 1.5
gravity = 14.715  # 9.81 * 1.5
jumpHeight = 45
maxStepSizeY = 2

disabledScenes = []

size = (1920, 1080)
unscaledSize = (192, 108)
center = (round(unscaledSize[0] / 2), round(unscaledSize[1] / 2))
fullscreen = False

assert size[0] / unscaledSize[0] == size[1] / unscaledSize[1]
scaleFactor = size[0] / unscaledSize[0]

dtype = np.float32

debug = False
showFps = True

character = True
characterAffectedByLight = False

physics = True
foliage = False

light = True
globalLights = True
showLightIntensityMap = False
raytracingSteps = 3
wallFalloff = .01 / raytracingSteps

camera = True
