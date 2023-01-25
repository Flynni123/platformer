import math

import numpy as np
from numba import cuda

import settings
import maths


@cuda.jit(device=True, fastmath=True)
def setPixel(surf, x, y, rgb):
    r, g, b = rgb
    r, g, b = math.floor(r), math.floor(g), math.floor(b)

    r = clamp(255, 0, r)
    g = clamp(255, 0, g)
    b = clamp(255, 0, b)

    x, y = int(x), int(y)
    surf[x][y] = 0x010000 * r + \
                 0x000100 * g + \
                 0x000001 * b


@cuda.jit(device=True, fastmath=True)
def addPixel(surf, x, y, rgb):
    r, g, b = rgb

    r2, g2, b2 = getRGB(surf, x, y)
    r += r2
    g += g2
    b += b2

    r, g, b = math.floor(r), math.floor(g), math.floor(b)

    r = clamp(255, 0, r)
    g = clamp(255, 0, g)
    b = clamp(255, 0, b)

    x, y = int(x), int(y)
    surf[x][y] = 0x010000 * r + \
                 0x000100 * g + \
                 0x000001 * b


@cuda.jit(device=True, fastmath=True)
def getRGB(surf, x, y):
    f = surf[x][y]
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

    return math.sqrt(math.pow(a_, 2) + math.pow(b_, 2))


@cuda.jit(device=True, fastmath=True)
def normalize(xy):
    x, y = xy
    vLength = distance(xy, (0, 0))
    return x / vLength, y / vLength


@cuda.jit(device=True, fastmath=True)
def dot(pos1, pos2):
    pos1normal = normalize(pos1)
    pos2normal = normalize(pos2)

    return pos1normal[0] * pos2normal[0] + pos1normal[1] * pos2normal[1]


@cuda.jit(device=True, fastmath=True)
def scalePos(pos_):
    return pos_[0] / settings.unscaledSize[0], \
           pos_[1] / settings.unscaledSize[0]


@cuda.jit(device=True, fastmath=True)
def clamp(top, bottom, x):
    out = x if x <= top else top
    out = out if out >= bottom else bottom
    return out


@cuda.jit(fastmath=True)
def _fragment(g0, comp, attr):
    ty: int = cuda.threadIdx.x
    tx: int = cuda.blockIdx.x

    if -1 < tx < g0.shape[0] and -1 < ty < g0.shape[1]:
        exposure = attr[0]
        blur = attr[1]
        scaleExposure = attr[2]

        setPixel(comp, tx, ty, (0, 0, 0))
        origin = getRGB(g0, tx, ty)

        # BLUR
        if blur > 0:

            bOutR = 0
            bOutG = 0
            bOutB = 0
            for x in range(-blur, blur+1):
                for y in range(-blur, blur+1):
                    c = getRGB(g0, tx + x, ty + y)
                    bOutR += c[0]
                    bOutG += c[1]
                    bOutB += c[2]

            area = ((blur * 2) + 1) ** 2
            origin = (bOutR / area,
                      bOutG / area,
                      bOutB / area)

        # EXPOSURE
        if exposure != 1:
            origin = (origin[0] * exposure,
                      origin[1] * exposure,
                      origin[2] * exposure)

        # SCALE EXPOSURE
        if scaleExposure >= 0:
            cScaleExposure = maths.getRGB(scaleExposure)

            origin = (origin[0] / cScaleExposure[0],
                      origin[1] / cScaleExposure[1],
                      origin[2] / cScaleExposure[2])

            origin = (origin[0] * 255,
                      origin[1] * 255,
                      origin[2] * 255)

        setPixel(comp, tx, ty, origin)

    cuda.syncthreads()


class shaderHandler:

    def __init__(self, size):
        self.run = False
        self.size = size
        self.comp = np.zeros(size, dtype=settings.dtype)

        self.first = True

    def fragment(self, g0: np.ndarray, attr: np.ndarray):
        if self.first:
            self.first = False
            with cuda.defer_cleanup():
                self.comp = g0
                _fragment[1024, 512](g0, self.comp, attr)
        else:
            with cuda.defer_cleanup():
                _fragment[1024, 512](g0, self.comp, attr)

        return self.comp
