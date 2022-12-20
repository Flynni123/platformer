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
    def __init__(self, size=settings.unscaledSize):
        self.size = size

        self.time = np.zeros(size)

        self.p = pg.Surface(self.size)
        self.p.fill((1, 255, 0))

    def reset(self):
        self.p.fill((1, 255, 0))

    def prepareForShader(self):
        return np.array([self.time, pg.surfarray.array2d(self.p)], dtype=settings.dtype)

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
        self.time = 0

        self.shaderHandler = shaders.shaderHandler(self.size)

    def update(self, ticks, pBuffer: PBuffer):
        self.t = ticks
        pBuffer.time += ticks

    def evaluate(self, pBuffer):

        self.composite = light.GBuffer(self.size)

        shaderInLines = []
        for l in self.lines:
            if isinstance(l, Line):
                shaderInLines.append(
                    l.prepareForShader()
                )

        shaderInLines = np.array(shaderInLines, dtype=np.float32)

        outLines = self.shaderHandler.fragment(shaderInLines)

        #self.lines = []
        for l in outLines:

            pairs = []
            for n in l:
                pairs.append((n[0], n[1]))

            color = l[0][3:6]
            normal = round(l[0][6]), round(l[0][7])
            noShadow = l[0][10]

            pg.draw.lines(self.composite.g0, color, False, pairs)
            pg.draw.lines(self.composite.g1, (normal[0], normal[1], 254*round(noShadow)), False, pairs)

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
