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
import keyboardSettings


class TimerError(Exception):
    pass


class SizeError(Exception):
    pass


class FallingError(Exception):
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

            smooth = maths.avg(self.tickList[-20:])
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
class Image:
    def __init__(self, gBuffer=light.GBuffer(), parallaxFac=1.0):
        self.gBuffer = gBuffer
        self.parallaxFac = parallaxFac
        self.image = self.gBuffer.g0


def loadImage(path, parallaxFac=1.0):
    p = os.path.abspath(path)

    image = pg.image.load(p).convert()

    if os.path.exists(f"assets/images/{os.path.split(p)[-1].split('.')[0]}g1.png"):
        normal = pg.image.load(f"assets/images/{os.path.split(p)[-1].split('.')[0]}g1.png")
        normal.convert()
        if settings.debug: print(f"loaded normal map {os.path.split(p)[-1].split('.')[0]}g1.png")
    else:
        normal = pg.Surface(image.get_size(), pg.SRCALPHA)

    if not image.get_size() == normal.get_size():
        raise SizeError("image and image normal not the same size")

    gBuffer = light.GBuffer(image.get_size(), image, normal)

    return Image(gBuffer, parallaxFac)


class Animation:
    def __init__(self, image: Image, timeOffset):
        self.__image = image.image
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

        return Image(light.GBuffer((self.offset, self.offset), out, pg.Surface((self.offset, self.offset))))


class AnimationHandler:

    def __init__(self, animations: dict):
        """
        walking
        standing
        running
        sitting
        """

        self.animations = animations

        if "walking" in self.animations:
            self.__walking = self.animations["walking"]
        else:
            raise NotImplementedError("missing animation")

        if "standing" in self.animations:
            self.__standing = self.animations["standing"]
        else:
            raise NotImplementedError("missing animation")

        if "running" in self.animations:
            self.__running = self.animations["running"]
        else:
            raise NotImplementedError("missing animation")

        if "sitting" in self.animations:
            self.__sitting = self.animations["sitting"]
        else:
            raise NotImplementedError("missing animation")

    def cTicks(self, t):
        for a in self.animations:
            self.animations[a].cTicks += t

    @property
    def walking(self):
        return self.__walking

    @property
    def standing(self):
        return self.__standing

    @property
    def running(self):
        return self.__running

    @property
    def sitting(self):
        return self.__sitting


class Character:

    def __init__(self, animations: AnimationHandler, rotation=0):
        self.animations = animations
        size = self.animations.standing.getImage().gBuffer.g0.get_size()

        self.pos = [(GetSystemMetrics(0) / 2 / settings.scaleFactor) - (size[0] / 2),
                    (GetSystemMetrics(1) / 2 / settings.scaleFactor) - (size[0] / 2)]
        self.velocity = maths.Vec2((0, 0))
        self.rotation = rotation
        self.floor = None

        self.jumping = False

        self.flipped = False
        self.current = self.animations.standing
        self.movementStart = 0
        self.oldOffset = 0

    def update(self, ticks, keys, floor: pg.mask.Mask, offset):
        # TODO: pBuffer
        self.animations.cTicks(ticks)

        if settings.physics:
            self.floor = floor
            mask = pg.mask.from_surface(self.current.getImage().gBuffer.g0)

            oldY = self.pos[1]
            y = 0

            while self.floor.overlap_area(mask, (round(self.pos[0] - offset), round(y))) == 0:
                y += 1

                if y > settings.unscaledSize[1]:
                    raise FallingError(f"y: {y} > maxY: {settings.unscaledSize[1]}")

            y -= 1
            floorY = y
            y = oldY - (self.velocity.y * ticks * settings.gravity)
            self.velocity.y -= settings.gravity * ticks

            if y < floorY:
                self.pos[1] = y
            else:
                self.pos[1] = floorY
                self.velocity.y = 0

        if keys[pg.K_a] == 1 or keys[pg.K_LEFT] == 1:
            self.movementStart += ticks

            if keys[keyboardSettings.run] == 1:
                self.velocity.x = -(settings.speed * settings.runFac * ticks * 100)
                self.current = self.animations.running
            else:
                self.velocity.x = -(settings.speed * ticks * 100)
                self.current = self.animations.walking

        elif keys[pg.K_d] == 1 or keys[pg.K_RIGHT] == 1:
            self.movementStart += ticks

            if keys[keyboardSettings.run] == 1:
                self.velocity.x = settings.speed * settings.runFac * ticks * 100
                self.current = self.animations.running
            else:
                self.velocity.x = settings.speed * ticks * 100
                self.current = self.animations.walking

        else:
            self.velocity.x = 0
            self.current = self.animations.standing
            self.movementStart = 0

        if keys[keyboardSettings.jump] == 1:
            if not self.jumping:
                self.jumping = True
                self.velocity.y += settings.jumpHeight / settings.gravity
        else:
            self.jumping = False

    def render(self, win: light.GBuffer):
        gb = self.current.getImage().gBuffer

        g1Mask = pg.mask.from_surface(gb.g0, 1)
        g1Mask.to_surface(gb.g1, setcolor=(0, 0, 255), unsetcolor=(0, 0, 0))

        if self.velocity.x > 0:
            win.blit(gb, self.pos)
            self.flipped = False
        elif self.velocity.x < 0:
            win.blit(gb.flip(), self.pos)
            self.flipped = True
        else:
            if self.flipped:
                win.blit(gb.flip(), self.pos)
            else:
                win.blit(gb, self.pos)

    def prepareForPBuffer(self, pBuffer: physics.PBuffer):
        mask = np.array(pg.surfarray.array2d(self.current.getImage().gBuffer.g0), dtype=settings.dtype)
        mask /= 256 ** 3
        mask = np.ceil(mask)
        r = 0x010000 * (128 + self.velocity.x)
        g = 0x000100 * (128 + self.velocity.y)
        mask *= r + g


class Foliage:

    def __init__(self, image: Union[Image, Animation], pos: maths.Vec2):
        self.image = image
        self.pos = pos

        if isinstance(self.image, Image):
            self.size = self.image.gBuffer.g0.get_size()
        else:
            self.size = self.image.getImage().gBuffer.g0.get_size()

        self.ticks = 0

    def update(self, ticks):
        self.ticks += ticks

        if isinstance(self.image, Animation):
            self.image.cTicks += ticks

    def render(self):
        if isinstance(self.image, Image):
            return self.image.gBuffer
        else:
            return self.image.getImage()


def renderWithOffset(win: light.GBuffer, src: Union[Image, light.GBuffer], offset: float, size, y=0):
    sizeX = size[0] / src.parallaxFac
    sx = size[0]

    if offset < -sx + settings.unscaledSize[0]:
        offset = -sx + settings.unscaledSize[0]
    elif offset > 0:
        offset = 0

    gBuffer = src if isinstance(src, light.GBuffer) else src.gBuffer

    win.blit(gBuffer, (round(offset * src.parallaxFac), y))
    win.blit(gBuffer, (round(offset * src.parallaxFac) + round(sizeX / src.parallaxFac), y))

    return offset


# scene
class SceneLayout:

    def __init__(self, Images, Lights: light.LightHandler, Physics: physics.PhysicsHandler, foliage: list, floor: Image):
        self.images = Images
        self.lights = Lights
        self.physic = Physics
        self.foliage = foliage
        self.floor = pg.mask.from_surface(floor.gBuffer.g0, 1)

        self.canvasSize = settings.unscaledSize


class Scene:

    def __init__(self, layout: SceneLayout, character: Character, enabled=False):
        self.character: Character = character
        self.images: list = layout.images
        self.physicsHandler: physics.PhysicsHandler = layout.physic
        self.floor = layout.floor
        self.foliage = layout.foliage

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
            if settings.character: self.character.update(ticks, keys, self.floor, self.offset)
            if settings.physics: self.physicsHandler.update(ticks, self.pBuffer)

            if keys[pg.K_a] == 1 or keys[pg.K_LEFT] == 1:

                if keys[keyboardSettings.run] == 1:
                    self.offset += settings.speed * settings.runFac * ticks * 100
                else:
                    self.offset += settings.speed * ticks * 100

            elif keys[pg.K_d] == 1 or keys[pg.K_RIGHT] == 1:

                if keys[keyboardSettings.run] == 1:
                    self.offset -= settings.speed * settings.runFac * ticks * 100
                else:
                    self.offset -= settings.speed * ticks * 100

    def render(self):
        if self.enabled:
            self.win.reset()

            for i in self.images:
                self.offset = renderWithOffset(self.win, i, self.offset, self.imageSize)

            if settings.physics:
                self.offset = renderWithOffset(self.win, self.physicsHandler.evaluate(self.pBuffer), self.offset,
                                               self.imageSize)

            if settings.foliage:
                for f in self.foliage:
                    self.offset = renderWithOffset(self.win, f.render(), self.offset, f.size)  # TODO: tree prevents offset from changing - fix

            if settings.character:
                self.character.render(self.win)

            if settings.light:
                self.win.blit(self.lightHandler.evaluate(self.win), (0, 0))

            if settings.character and not settings.characterAffectedByLight:
                self.character.render(self.win)

            print(self.offset)

            return pg.transform.scale(self.win.g0, (math.floor(self.win.g0.get_size()[0] * settings.scaleFactor),
                                                    settings.size[1]))
        else:
            return pg.Surface((0, 0))
