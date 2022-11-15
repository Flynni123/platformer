import math

from numba import cuda
import numpy as np
import settings


@cuda.jit(device=True, fastmath=True)
def setPixel(surf, x, y, rgba):
    r, g, b, a = rgba
    r, g, b, a = math.floor(r), math.floor(g), math.floor(b), math.floor(a)
    surf[y][x] += 0x01000000 * r + \
                  0x00010000 * g + \
                  0x00000100 * b + \
                  0x00000001 * a


@cuda.jit(device=True, fastmath=True)
def getRGBA(surf, x, y):
    f = surf[y][x]
    r = math.floor(f / 0x01000000)
    f -= r * 0x01000000
    g = math.floor(f / 0x00010000)
    f -= g * 0x00010000
    b = math.floor(f / 0x00000100)
    f -= b * 0x00000100
    a = math.floor(f / 0x00000001)
    f -= a * 0x00000001
    return r, g, b, a


@cuda.jit(device=True, fastmath=True)
def distance(pos1, pos2):
    a_ = pos1[0] - pos2[0]
    b_ = pos1[1] - pos2[1]

    return abs(math.pow(math.pow(a_, 2) + math.pow(b_, 2), .5))


@cuda.jit(device=True, fastmath=True)
def normalize(x, y):
    vLength = abs(math.pow(math.pow(x, 2) + math.pow(y, 2), .5))
    return y / vLength, y / vLength


@cuda.jit(fastmath=True)
def _fragment(lights, lSurfs, comp, g1):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    def scalePos(pos_):
        return pos_[0] / settings.unscaledSize[0], \
               pos_[1] / settings.unscaledSize[0]

    cuda.syncthreads()
    if tx + ty * bw < comp.size:

        for i in range(len(lSurfs)):
            l = lights[i]
            s = lSurfs[i]
            fIntensity = l[2] * ((1 - distance(scalePos((tx, ty)), scalePos((l[0], l[1])))) ** 2)
            setPixel(s, tx, ty, (l[4] * fIntensity,
                                 l[5] * fIntensity,
                                 l[6] * fIntensity,
                                 l[7] * fIntensity))


def fragment(sizex, sizey, lights, lSurfs, comp, g1):
    _fragment[sizey, sizex](lights, lSurfs, comp, g1)
