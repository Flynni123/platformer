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


"""
void line(int x0, int y0, int x1, int y1)
{
    int dx =  abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2; /* error value e_xy */

    while (1) {
        setPixel(x0, y0);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 > dy) { err += dy; x0 += sx; } /* e_xy+e_x > 0 */
        if (e2 < dx) { err += dx; y0 += sy; } /* e_xy+e_y < 0 */
    }
}
"""


@cuda.jit(fastmath=True)
def _fragment(lines, pBuffer):
    ty = cuda.threadIdx.x
    tx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw

    if -1 < tx < pBuffer.shape[0] and -1 < ty < pBuffer.shape[1]:

        c = getRGB(pBuffer, tx, ty)
        velocity = (128 - c[0], 128 - c[1])

    cuda.syncthreads()


class shaderHandler:
    def __init__(self, size):
        self.run = False
        self.size = size

    @staticmethod
    def fragment(lines: np.ndarray, pBuffer):

        _fragment[1024, 512](lines, pBuffer)

        return lines
