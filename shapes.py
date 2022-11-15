import math
import pygame as pg
from numba import cuda
import maths


# r channel - 128: normal x
# g channel - 128: normal y


@cuda.jit
def fragment(comp):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    def setPixel(surf, rgba):
        x, y = tx, ty
        r, g, b, a = rgba
        r, g, b, a = math.floor(r), math.floor(g), math.floor(b), math.floor(a)
        surf[y][x] += 0x01000000 * r + \
                      0x00010000 * g + \
                      0x00000100 * b + \
                      0x00000001 * a

    def distance(pos1, pos2):
        a_ = pos1[0] - pos2[0]
        b_ = pos1[1] - pos2[1]

        return abs(pow(pow(a_, 2) + pow(b_, 2), .5))

    def normalize():
        vLength = abs(pow(pow(tx, 2) + pow(ty, 2), .5))
        return tx/vLength, ty/vLength

    if tx + ty * bw < comp.size:
        pass


class Circle:
    def __init__(self, radius, color):
        self.g0 = pg.Surface((radius * 2, radius * 2), pg.SRCALPHA)
        self.g1 = pg.Surface((radius * 2, radius * 2), pg.SRCALPHA)
