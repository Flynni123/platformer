import math
import light
import pygame as pg
import time
import numpy as np
import os
from win32api import GetSystemMetrics
import json
import settings
import maths


class TimerError(Exception):
    pass


class Character:

    def __init__(self, image: pg.Surface, rotation=0, lockX=True):

        self.image = image

        self.pos = maths.Vec2((GetSystemMetrics(0) / 2, 0), rotation)
        self.rotation = rotation
        self.lockX = lockX

        screenX = GetSystemMetrics(0)
        pos = round(screenX / 2 + self.pos.x / 2)
        self.pos.x = pos

    def update(self, ticks, keys):
        # movement
        pass

    def render(self):
        return pg.transform.rotate(self.image, self.pos.a), self.pos.pos


# time
class TimeHandler:

    def __init__(self):
        self.s = 1
        self._start = 0

    def start(self):
        self._start = time.time()

    def getTicks(self):
        if self._start > 0:
            out = (time.time() - self._start) * self.s
            self._start = time.time()
            return out
        else:
            raise TimerError("didnt start()")

    def setSpeed(self, num: float):
        """
        :param num: speed of the game engine. 1 equals one second per second
        """
        self.s = num


# images
class Image:
    def __init__(self, path):
        p = os.path.abspath(path)
        self.image = pg.image.load(p)
        with open("assets/images/images.json", "r") as f:
            self.parallaxFac = json.load(f)[p.split("\\")[-1]]


class ImageHandler:
    def __init__(self, images: list):
        self.images = []
        for i in images:
            self.images.append(Image(i))


def renderWithOffset(win: pg.Surface, src: Image, offset: float, y=0):
    sizex = src.image.get_size()[0]
    if offset > sizex:
        offset -= sizex
    elif offset < -sizex:
        offset += sizex

    win.blit(src.image, (round(offset * src.parallaxFac), y))
    win.blit(src.image, (round(offset * src.parallaxFac) + round(sizex / src.parallaxFac), y))
    win.blit(src.image, (round(offset * src.parallaxFac) - round(sizex / src.parallaxFac), y))


# scene
class SceneLayout:

    def __init__(self, Images: ImageHandler, Lights: light.LightHandler):
        self.lights = Lights

        self.images = Images.images


class Scene:

    def __init__(self, layout: SceneLayout, character, enabled=False):
        self.character = character
        self.images = layout.images
        self.lightHandler = layout.lights

        self.offset = 0
        self.surf = pg.Surface(settings.canvasSize, pg.SRCALPHA)

        self.enabled = enabled

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def update(self, ticks, keys):

        if keys[pg.K_a] == 1 or keys[pg.K_LEFT] == 1:
            self.offset += settings.speed * ticks
        elif keys[pg.K_d] == 1 or keys[pg.K_RIGHT] == 1:
            self.offset += -settings.speed * ticks

    def render(self, upscaled=True):
        self.surf.fill((0, 0, 0))
        for i in self.images:
            renderWithOffset(self.surf, i, self.offset)

        self.surf.blit(self.lightHandler.evaluate(self.surf), (0, 0))

        return self.surf if not upscaled else pg.transform.scale(self.surf, (math.floor(self.surf.get_size()[0]*settings.scaleFactor), settings.size[1]))
