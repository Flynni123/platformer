from enum import Enum
from typing import *

import numpy as np
import pygame as pg

import maths
import light
import physics_shaders as shaders
import shapes
import settings
import colors


class Node:
    def __init__(self, pos, fixed=False):
        self.pos = pos
        self.fixed = fixed


class Line:
    def __init__(self, nodes: List[Node], color=(255, 255, 255)):
        self.nodes = nodes
        self.color = color

    def prepareForShader(self):

        out = []
        for n in self.nodes:
            out.append([n.pos[0], n.pos[1], 1 if n.fixed else 0, self.color[0], self.color[1], self.color[2]])

        out = np.array(out, dtype=settings.dtype)
        return out


def line(startPos, color=(255, 255, 255), length=10):
    pos = maths.Vec2(startPos)

    n = []
    for i in range(length):
        n.append(Node((pos.x, pos.y+10*i)))
    n[0].fixed = True

    return Line(n, color)


class PBuffer:
    def __init__(self, size=settings.unscaledSize, p=None):
        self.size = size

        if p is None: self.p = pg.Surface(self.size, pg.SRCALPHA)
        else: self.p = p

    def reset(self):
        self.p = pg.Surface(self.size)

    def prepareForShader(self):
        return np.array(pg.surfarray.array2d(self.p), dtype=settings.dtype)

    def blit(self, pBuffer, dest):
        assert isinstance(pBuffer, PBuffer)

        pBuffer.p.convert()

        self.p.blit(pBuffer.p, dest)

    def upscale(self, fac=settings.scaleFactor):
        pos = (round(self.p.get_size()[0] * fac), round(self.p.get_size()[1] * fac))
        self.p = pg.transform.scale(self.p, pos)

    def downscale(self, fac=settings.scaleFactor):
        pos = (round(self.p.get_size()[0] / fac), round(self.p.get_size()[1] / fac))
        self.p = pg.transform.scale(self.p, pos)

    def __copy__(self):
        g = PBuffer(self.size)
        g.p = self.p
        return g


class PhysicsLayout:
    def __init__(self, objects: list):
        self.objects = objects
        self.p = None

    def run(self):
        self.p = PhysicsHandler(self.objects)


class PhysicsHandler:

    def __init__(self, Lines: list):
        self.lines = Lines
        self.size = settings.unscaledSize

        self.c = pg.Surface(settings.unscaledSize)
        self.composite = light.GBuffer(self.size)
        self.t = 0

        self.shaderHandler = shaders.shaderHandler(self.size)

    def update(self, ticks):
        self.t = ticks

    def evaluate(self, pBuffer):

        self.composite = light.GBuffer(self.size)

        shaderInLines = []
        for l in self.lines:
            if isinstance(l, Line):
                shaderInLines.append(
                    l.prepareForShader()
                )

        shaderInLines = np.array(shaderInLines, dtype=np.float32)
        shaderInP = pBuffer.prepareForShader()

        out = self.shaderHandler.fragment(shaderInLines, shaderInP)

        #self.lines = []
        for l in out:
            for e, n in enumerate(l):
                pos = (n[0], n[1])
                fixed = bool(round(n[2]))
                color = n[3:]

                if not e == 0:
                    pair = (pos, (l[e-1][0], l[e-1][1]))
                    pg.draw.line(self.composite.g0, color, pair[0], pair[1])
                    pg.draw.line(self.composite.g1, (120, 120, 0), pair[0], pair[1])

        return self.composite


class test:
    clock = pg.time.Clock()
    pg.init()
    pg.display.init()

    def __init__(self):

        self.disp = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        self.win = light.GBuffer(self.disp.get_size())
        self.size = self.disp.get_rect().size

        l = [line((10, 10), length=5), line((100, 10), length=5)]

        ph = PhysicsHandler(l)
        pBuffer = PBuffer()

        self.run = True
        while self.run:

            self.win.reset()
            pu = ph.evaluate(pBuffer=pBuffer)
            pu.upscale()
            self.win.blit(pu, (0, 0))

            for event in pg.event.get():

                if event.type == pg.QUIT:
                    self.run = False

                elif event.type == pg.KEYDOWN:

                    if event.key == pg.K_ESCAPE:
                        self.run = False

            self.disp.blit(self.win.g0, (0, 0))

            pg.display.update()
            self.clock.tick()
            if not self.run:
                print(round(self.clock.get_fps()))


if __name__ == '__main__':
    test()
