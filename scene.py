import json
import math
import os
import time
from typing import *

import pygame as pg
from win32api import GetSystemMetrics
import numpy as np

import colors
import light
import maths
import physics
import settings


class TimerError(Exception):
    pass


class SizeError(Exception):
    pass


# time
class TimeHandler:

    def __init__(self):

        self.s = 1
        self._start = 0
        self.error = 0
        self.tickList = []
        self.smoothList = []

    def start(self):
        self._start = time.time()

    def getTicks(self, fps):
        if self._start > 0:
            out = (time.time() - self._start) * self.s
            self._start = time.time()
            self.tickList.append(out)

            smooth = maths.avg(self.tickList[-10:])
            if settings.debug: self.smoothList.append(smooth)

            self.error += out - smooth
            if self.error >= 1:
                smooth /= 10

            return smooth
        else:
            raise TimerError("didnt start()")

    def setSpeed(self, num: float):
        """
        :param num: speed of the game engine. 1 equals one second per second
        """
        self.s = num


# images
class ImageNoPath:
    def __init__(self, gBuffer, parallaxFac):
        self.gBuffer = gBuffer
        self.parallaxFac = parallaxFac
        self.image = self.gBuffer.g0


class Image:
    def __init__(self, path, parallaxFac):
        p = os.path.abspath(path)
        self.parallaxFac = parallaxFac

        self.image = pg.image.load(p).convert()

        if os.path.exists(f"assets/images/{os.path.split(p)[-1].split('.')[0]}g1.png"):
            self.normal = pg.image.load(f"assets/images/{os.path.split(p)[-1].split('.')[0]}g1.png")
            self.normal.convert()
            if settings.debug: print(f"loaded normal map {os.path.split(p)[-1].split('.')[0]}g1.png")
        else:
            self.normal = pg.Surface(self.image.get_size(), pg.SRCALPHA)
            # pg.mask.from_surface(self.image, 1).to_surface(unsetcolor=(0, 0, 0), setcolor=(0, 0, 0))

        if not self.image.get_size() == self.normal.get_size():
            raise SizeError("image and image normal not the same size")

        self.gBuffer = light.GBuffer(self.image.get_size(), self.image, self.normal)


class Animation:
    def __init__(self, image: str, timeOffset):
        self.__image = pg.image.load(image).convert()
        self.offset = self.__image.get_height()
        self.timeOffset = timeOffset

        sizeX = self.__image.get_width()

        self.c = 0
        self.cTicks = 0

        self.subImages = []
        for i in range(int((sizeX / self.offset))):
            rect = pg.Rect((self.offset * i, 0), (self.offset, self.offset))
            subI = self.__image.subsurface(rect)
            self.subImages.append(subI)

    def getImage(self):

        if self.cTicks >= self.timeOffset:
            self.c += 1
            self.cTicks = 0

            if self.c >= len(self.subImages):
                self.c = 0

        out = self.subImages[self.c].convert()

        return light.GBuffer((self.offset, self.offset), out, pg.Surface((self.offset, self.offset)))


class Character:

    def __init__(self, animation: Animation, rotation=0, lockX=True):
        self.animation = animation

        self.pos = [(GetSystemMetrics(0) / 2 / settings.scaleFactor) - 6,
                    (GetSystemMetrics(1) / 2 / settings.scaleFactor) - 6]
        self.velocity = maths.Vec2((0, 0))
        self.rotation = rotation
        self.lockX = lockX

    def update(self, ticks, keys, pBuffer):
        # TODO: movement
        # TODO: pBuffer
        self.animation.cTicks += ticks

    def render(self):
        return self.animation.getImage()

    def prepareForPBuffer(self, pBuffer: physics.PBuffer):
        mask = np.array(pg.surfarray.array2d(self.render().g0), dtype=settings.dtype)
        mask /= 256**3
        mask = np.ceil(mask)
        r = 0x010000 * (128 + self.velocity.x)
        g = 0x000100 * (128 + self.velocity.y)
        mask *= r + g


def renderWithOffset(win: light.GBuffer, src: Union[Image, ImageNoPath, light.GBuffer], offset: float, size, y=0):
    sizeX = size[0] / src.parallaxFac
    sx = size[0]

    if offset < -sx+settings.unscaledSize[0]:
        offset = -sx+settings.unscaledSize[0]
    elif offset > 0:
        offset = 0

    gBuffer = src if isinstance(src, light.GBuffer) else src.gBuffer

    win.blit(gBuffer, (round(offset * src.parallaxFac), y))
    win.blit(gBuffer, (round(offset * src.parallaxFac) + round(sizeX / src.parallaxFac), y))

    return offset


# scene
class SceneLayout:

    def __init__(self, Images, Lights: light.LightHandler, Physics: physics.PhysicsHandler):
        self.images = Images
        self.lights = Lights
        self.physic = Physics

        self.canvasSize = settings.unscaledSize


class Scene:

    def __init__(self, layout: SceneLayout, character: Character, enabled=False):
        self.character: Character = character
        self.images: list = layout.images
        self.physicsHandler: physics.PhysicsHandler = layout.physic

        self.lightHandler: light.LightHandler = layout.lights
        self.canvasSize = layout.canvasSize
        self.imageSize = self.images[0].image.get_size()

        self.offset = 0
        self.win = light.GBuffer(self.canvasSize)
        self.pBuffer = physics.PBuffer(self.canvasSize)

        self.enabled = enabled

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def update(self, ticks, keys):
        if self.enabled:
            if settings.character: self.character.update(ticks, keys, self.pBuffer)
            if settings.physics: self.physicsHandler.update(ticks)

            if keys[pg.K_a] == 1 or keys[pg.K_LEFT] == 1:
                self.offset += settings.speed * ticks * 100
            elif keys[pg.K_d] == 1 or keys[pg.K_RIGHT] == 1:
                self.offset += -settings.speed * ticks * 100

    def render(self):
        if self.enabled:
            self.win.reset()

            for i in self.images:
                self.offset = renderWithOffset(self.win, i, self.offset, self.imageSize)

            if settings.physics:
                self.offset = renderWithOffset(self.win, self.physicsHandler.evaluate(self.pBuffer), self.offset, self.imageSize)

            if settings.character:
                self.win.blit(self.character.render(), self.character.pos)

            if settings.light:
                self.win.blit(self.lightHandler.evaluate(self.win), (0, 0))

            return pg.transform.scale(self.win.g0, (math.floor(self.win.g0.get_size()[0] * settings.scaleFactor),
                                                    settings.size[1]))
        else:
            return pg.Surface((0, 0))


class test:
    clock = pg.time.Clock()
    pg.init()
    pg.display.init()

    def __init__(self):

        self.win = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        self.size = self.win.get_rect().size

        th = TimeHandler()
        c = Character(Animation("assets/images/character.png", 1))

        firstRun = True

        self.run = True
        while self.run:

            if firstRun:
                firstRun = False
                th.start()

            self.win.fill((0, 0, 0))

            c.update(th.getTicks(self.clock.get_fps()), [], physics.PBuffer())
            a = c.render()
            self.win.blit(a, c.pos)

            for event in pg.event.get():

                if event.type == pg.QUIT:
                    self.run = False

                elif event.type == pg.KEYDOWN:

                    if event.key == pg.K_ESCAPE:
                        self.run = False

            pg.display.update()
            self.clock.tick()
            x = self.clock.get_fps()
            if not self.run:
                print(round(x))


if __name__ == '__main__':
    test()
