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
def multPixel(surf, x, y, rgb):
    r, g, b = rgb
    r /= 255
    g /= 255
    b /= 255

    x, y = int(x), int(y)
    r2, g2, b2 = getRGB(surf, x, y)
    r2 /= 255
    g2 /= 255
    b2 /= 255

    r = clamp(255, 0, r)
    g = clamp(255, 0, g)
    b = clamp(255, 0, b)

    r2 = clamp(255, 0, r)
    g2 = clamp(255, 0, g)
    b2 = clamp(255, 0, b)

    r *= r2
    g *= g2
    b *= b2

    r *= 255
    g *= 255
    b *= 255

    r, g, b = math.floor(r), math.floor(g), math.floor(b)
    surf[x, y] = 0x010000 * r + \
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
def _fragment(lights, g0, g1, comp, attr):
    ty: int = cuda.threadIdx.x
    tx: int = cuda.blockIdx.x

    if 0 <= tx < g0.shape[0] and 0 <= ty < g0.shape[1]:

        comp[tx, ty] = 0

        g0r, g0g, g0b = getRGB(g0, tx, ty)
        g1r, g1g, g1b = getRGB(g1, tx, ty)

        # --- LIGHT ---

        normalVec = (g1r - 128, g1g - 128)
        vec = (tx, ty)

        for l in lights:

            match round(l[0]):

                case 0:  # pointLight

                    lightDir = (l[1] - tx, l[2] - ty)
                    lightVec = (l[1], l[2])

                    scalar = clamp(1, 0, dot(lightDir, normalVec))
                    scalar = 1 if g1r + g1g == 0 else scalar
                    scalar = 1 if g1b == 255 else scalar

                    fIntensity = l[3] * \
                                 math.pow((1 - (distance(scalePos(vec), scalePos(lightVec)) / l[8])), 2) * \
                                 scalar  # final intensity (intensity * falloff * normalFalloff)

                    addPixel(comp, tx, ty, (l[5] * fIntensity * (g0r / 255),
                                            l[6] * fIntensity * (g0g / 255),
                                            l[7] * fIntensity * (g0b / 255)))

                    if g0[tx, ty] == 0:
                        addPixel(comp, tx, ty, (l[5] * l[4] * fIntensity,
                                                l[6] * l[4] * fIntensity,
                                                l[7] * l[4] * fIntensity))

                case 1:  # spotLight

                    lightDir = (l[1] - tx, l[2] - ty)
                    lightVec = (l[1], l[2])

                    scalar = clamp(1, 0, dot(lightDir, normalVec))
                    scalar = 1 if g1r + g1g == 0 else scalar
                    scalar = 1 if g1b == 255 else scalar

                    angelToLight = round(maths.angelTo(lightVec, vec), 1)
                    a = angelToLight - l[9]
                    aIntensity = 1 if -(l[10] / 2) < a < (l[10] / 2) else 0

                    if aIntensity == 0:
                        if -(l[10] / 2) - 1 < a:
                            aIntensity = clamp(1, 0, a - math.floor(a))

                    fIntensity = l[3] * \
                                 math.pow((1 - (distance(scalePos(vec), scalePos(lightVec)) / l[8])), 2) * scalar * aIntensity

                    addPixel(comp, tx, ty, (l[5] * fIntensity * (g0r / 255),
                                            l[6] * fIntensity * (g0g / 255),
                                            l[7] * fIntensity * (g0b / 255)))

                    if g0[tx, ty] == 0:
                        addPixel(comp, tx, ty, (l[5] * l[4] * fIntensity,
                                                l[6] * l[4] * fIntensity,
                                                l[7] * l[4] * fIntensity))

                case 2:  # globalLight

                    addPixel(comp, tx, ty, (l[5] * (g0r / 255) * l[3],
                                            l[6] * (g0g / 255) * l[3],
                                            l[7] * (g0b / 255) * l[3]))

        # --- CAMERA ---

        exposure = attr[0]
        blur = attr[1]

        origin = getRGB(comp, tx, ty)

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

        cuda.syncthreads()
        setPixel(comp, tx, ty, origin)

    else:
        cuda.syncthreads()


class shaderHandler:
    """
    g1 blue channel:    254: no shadow
                        255: no normals

    g1 red and green channels: normal vectors (x+128, y+128)

    g0: visible screen
    """

    def __init__(self, size):
        self.size = size
        self.comp = np.zeros(size, dtype=settings.dtype)

        self.l = np.array([], dtype=settings.dtype)
        self.g0 = np.array([], dtype=settings.dtype)
        self.g1 = np.array([], dtype=settings.dtype)
        self.attr = np.array([], dtype=settings.dtype)

    def setAttributes(self, g0=None, g1=None, l=None, attr=None):
        if g0 is not None:
            self.g0 = g0

        if g1 is not None:
            self.g1 = g1

        if l is not None:
            self.l = l

        if attr is not None:
            self.attr = attr

    def fragment(self):
        with cuda.defer_cleanup():
            _fragment[1024, 512](self.l, self.g0, self.g1, self.comp, self.attr)

    def getResult(self):
        return self.comp.round(0).astype(np.int32)
