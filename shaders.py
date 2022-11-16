import math

import numpy as np
from numba import cuda

import settings


@cuda.jit(device=True, fastmath=True)
def setPixel(surf, x, y, rgb):
    r, g, b = rgb
    r = 255 if r > 255 else r
    g = 255 if g > 255 else g
    b = 255 if b > 255 else b

    r = 0 if r < 0 else r
    g = 0 if g < 0 else g
    b = 0 if b < 0 else b
    r, g, b = math.floor(r), math.floor(g), math.floor(b)
    surf[y][x] = 0x010000 * r + \
                 0x000100 * g + \
                 0x000001 * b


@cuda.jit(device=True, fastmath=True)
def addPixel(surf, x, y, rgb):
    r, g, b = rgb

    r2, g2, b2 = getRGB(surf, x, y)
    r += r2
    g += g2
    b += b2

    r = 255 if r > 255 else r
    g = 255 if g > 255 else g
    b = 255 if b > 255 else b
    r = 0 if r < 0 else r
    g = 0 if g < 0 else g
    b = 0 if b < 0 else b

    r, g, b = math.floor(r), math.floor(g), math.floor(b)
    surf[y][x] = 0x010000 * r + \
                 0x000100 * g + \
                 0x000001 * b


@cuda.jit(device=True, fastmath=True)
def getRGB(surf, x, y):
    f = surf[y][x]
    r = math.floor(f / 0x010000)
    f -= r * 0x010000
    g = math.floor(f / 0x000100)
    f -= g * 0x000100
    b = math.floor(f / 0x000001)
    f -= b * 0x000001
    return r, g, b


@cuda.jit(device=True, fastmath=True)
def distance(pos1, pos2):
    a_ = pos1[0] - pos2[0]
    b_ = pos1[1] - pos2[1]

    return abs(math.pow(math.pow(a_, 2) + math.pow(b_, 2), .5))


@cuda.jit(device=True, fastmath=True)
def fromPolar(r, roh):
    degRoh = roh * (math.pi / 180)
    return r * math.sin(degRoh), - (r * math.cos(degRoh))


@cuda.jit(device=True, fastmath=True)
def angleTo(pos1, pos2):
    a_ = pos1[0] - pos2[0]
    b_ = pos1[1] - pos2[1]
    return math.degrees(math.atan2(-a_, b_))


@cuda.jit(device=True, fastmath=True)
def normalize(xy):
    x, y = xy
    vLength = distance((0, 0), xy)
    return x / vLength, y / vLength


@cuda.jit(device=True, fastmath=True)
def scalePos(pos_):
    return pos_[0] / settings.unscaledSize[0], \
           pos_[1] / settings.unscaledSize[0]


@cuda.jit(fastmath=True)
def _fragment(lights, g0, g1, comp):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    if 0 < tx < g0.shape[1] and 0 < ty < g0.shape[0]:

        for i in range(len(lights)):
            l = lights[i]

            g0r, g0g, g0b = getRGB(g0, tx, ty)
            g1r, g1g, g1b = getRGB(g1, tx, ty)

            normalVec = (g1r - 128, g1g - 128)
            lightVec = (l[0], l[1])
            vec = (tx, ty)

            # TODO: add normals (dot)

            fIntensity = l[2] * math.pow((1 - (distance(scalePos(vec), scalePos(lightVec))/l[7])), 2)

            addPixel(comp, tx, ty, (l[4] * fIntensity * (g0r/255),
                                    l[5] * fIntensity * (g0g/255),
                                    l[6] * fIntensity * (g0b/255)))
            if g0r + g0g + g0b == 0:
                addPixel(comp, tx, ty, (l[4] * fIntensity * l[3],
                                        l[5] * fIntensity * l[3],
                                        l[6] * fIntensity * l[3]))


class shaderHandler:
    def __init__(self, size):
        self.run = False
        self.size = size

    def fragment(self, l, g0, g1):
        comp = np.zeros((self.size[1], self.size[0]))
        _fragment[1024, 512](l, g0, g1, comp)
        return comp
