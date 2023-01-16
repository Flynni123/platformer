from typing import *

import numpy as np
import pygame as pg
from pygame import gfxdraw

import light
import maths
import physics_shaders as shaders
import settings


class Node:
    def __init__(self, pos, velocity=(0, 0), fixed=False):
        self.pos = pos
        self.v = velocity
        self.fixed = fixed


class Line:
    def __init__(self, nodes: List[Node], color=(255, 255, 255), normal=(0, 0), noShadow=True):
        self.nodes = nodes
        self.color = color
        self.normal = normal
        self.noShadow = noShadow

    def prepareForShader(self):

        out = []
        for n in self.nodes:
            out.append([n.pos[0], n.pos[1], 1 if n.fixed else 0,
                        self.color[0], self.color[1], self.color[2],
                        self.normal[0]+128, self.normal[1]+128,
                        n.v[0], n.v[1], 1 if self.noShadow else 0])

        out = np.array(out, dtype=settings.dtype)
        return out


def line(startPos, color=(255, 255, 255), length=10, normal=(-10, -10)):
    pos = maths.Vec2(startPos)

    n = []
    for i in range(length):
        n.append(Node((pos.x, pos.y+10*i)))
    n[0].fixed = True

    return Line(n, color, normal)


class PBuffer:
    def __init__(self, size=settings.unscaledSize,  p=None):
        self.size = size

        self.time = np.zeros(size)

        if p is None:
            self.p = pg.Surface(self.size)
            self.p.fill((128, 128, 0))
        else:
            self.p = p

    def reset(self):
        self.p.fill((1, 255, 0))

    def prepareForShader(self):
        return np.array([self.time, pg.surfarray.array2d(self.p)], dtype=settings.dtype)

    def blit(self, pBuffer, pos):
        assert isinstance(pBuffer, PBuffer)

        pBuffer.p.convert()

        self.p.blit(pBuffer.p, pos)

    def upscale(self, fac=settings.scaleFactor):
        pos = (round(self.p.get_size()[0] * fac), round(self.p.get_size()[1] * fac))
        self.p = pg.transform.scale(self.p, pos)

    def downscale(self, fac=settings.scaleFactor):
        pos = (round(self.p.get_size()[0] / fac), round(self.p.get_size()[1] / fac))
        self.p = pg.transform.scale(self.p, pos)

    def __copy__(self):
        g = PBuffer(self.size, self.p)
        return g


class PhysicsHandler:

    def __init__(self, Lines: list):
        self.lines = Lines
        self.size = settings.unscaledSize

        self.c = pg.Surface(settings.unscaledSize)
        self.composite = light.GBuffer(self.size)
        self.t = 0
        self.time = 0

        self.shaderHandler = shaders.shaderHandler(self.size)

    def update(self, ticks, pBuffer: PBuffer):
        self.t = ticks
        pBuffer.time += ticks

    def evaluate(self, pBuffer):
        # assert isinstance(pBuffer, PBuffer)

        self.composite = light.GBuffer(self.size)

        shaderInLines = []
        for l in self.lines:
            if isinstance(l, Line):
                shaderInLines.append(
                    l.prepareForShader()
                )

        shaderInLines = np.array(shaderInLines, dtype=settings.dtype)

        outLines = self.shaderHandler.fragment(shaderInLines, self.t)

        for l in outLines:

            pairs = []
            for n in l:
                pairs.append((n[0], n[1]))

            color = l[0][3:6]
            normal = round(l[0][6]), round(l[0][7])
            noShadow = l[0][10]

            pairs = np.array(pairs)
            pairs = pairs.round(0).astype(np.int32)

            for p in range(len(pairs) - 1):
                gfxdraw.line(self.composite.g0, pairs[p][0], pairs[p][1], pairs[p + 1][0], pairs[p + 1][1], color)
                gfxdraw.line(self.composite.g1, pairs[p][0], pairs[p][1], pairs[p + 1][0], pairs[p + 1][1], (normal[0], normal[1], 254*round(noShadow)))

        return self.composite
