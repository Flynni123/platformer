import math

import numpy as np
import pygame as pg
from numba import cuda

import settings


@cuda.jit(device=True, fastmath=True)
def setPixel(surf, x, y, rgb):
    r, g, b = rgb
    r, g, b = math.floor(r), math.floor(g), math.floor(b)

    r = 255 if r > 255 else r
    g = 255 if g > 255 else g
    b = 255 if b > 255 else b

    r = 0 if r < 0 else r
    g = 0 if g < 0 else g
    b = 0 if b < 0 else b

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

    r = 255 if r > 255 else r
    g = 255 if g > 255 else g
    b = 255 if b > 255 else b
    r = 0 if r < 0 else r
    g = 0 if g < 0 else g
    b = 0 if b < 0 else b

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


@cuda.jit(fastmath=True)
def _fragment(lights, g0, g1, comp):
    ty = cuda.threadIdx.x
    tx = cuda.blockIdx.x

    if -1 < tx < g0.shape[0] and -1 < ty < g0.shape[1]:

        for l in lights:

            g0r, g0g, g0b = getRGB(g0, tx, ty)
            g1r, g1g, g1b = getRGB(g1, tx, ty)

            normalVec = (g1r - 128, g1g - 128)
            lightDir = (l[0]-tx, l[1]-ty)
            lightVec = (l[0], l[1])
            vec = (tx, ty)

            scalar = dot(lightDir, normalVec)
            scalar = 0 if scalar < 0 else scalar
            scalar = 1 if scalar > 1 else scalar

            scalar = 1 if g1r + g1g == 0 else scalar
            scalar = 1 if g1b == 255 else scalar

            fIntensity = l[2] * math.pow((1 - (distance(scalePos(vec), scalePos(lightVec)) / l[7])), 2) * scalar

            dx, dy = l[0] - tx, l[1] - ty
            dst = distance(lightVec, (tx, ty))

            step = (dx / dst, dy / dst)

            x, y = tx, ty
            running = 0

            if g0[tx, ty] == 0:

                for point in range(round(dst)):

                    if not g0[int(x), int(y)] == 0:
                        if not getRGB(g1, int(x), int(y))[2] == 254:
                            running += 1

                    x += step[0]
                    y += step[1]

            if running == 0:
                addPixel(comp, tx, ty, (l[4] * fIntensity * (g0r / 255),
                                        l[5] * fIntensity * (g0g / 255),
                                        l[6] * fIntensity * (g0b / 255)))

                if g0r + g0g + g0b == 0:
                    addPixel(comp, tx, ty, (l[4] * fIntensity * l[3],
                                            l[5] * fIntensity * l[3],
                                            l[6] * fIntensity * l[3]))

    cuda.syncthreads()


class shaderHandler:
    """
    g1 blue channel:    254: no shadow
                        255: no normals

    g1 red and green channels: normal vectors

    g0: visible screen
    """
    def __init__(self, size):
        self.run = False
        self.size = size
        self.comp = np.zeros(size, dtype=settings.dtype)

    def fragment(self, l: np.ndarray, g0: np.ndarray, g1: np.ndarray):

        self.comp.fill(0)
        with cuda.defer_cleanup():
            _fragment[1024, 512](l, g0, g1, self.comp)

        return self.comp
