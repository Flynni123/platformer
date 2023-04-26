import math
import os
import time
from typing import *

import pygame as pg
from win32api import GetSystemMetrics

import colors
import camera
import keyboardSettings
import light
import maths
import physics
import settings
import shaders


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
        self.tickList = []

    def start(self):
        self._start = time.time()

    def getTicks(self):
        if self._start > 0:
            out = (time.time() - self._start) * self.s
            self._start = time.time()
            self.tickList.append(out)

            smooth = maths.avg(self.tickList[-20:])
            self.tickList = self.tickList[-20:]

            return smooth
        else:
            raise TimerError("did\'nt start()")

    @property
    def speed(self):
        return self.s

    @speed.setter
    def speed(self, s):
        self.s = s


# images
class Image:
    def __init__(self, gBuffer=light.GBuffer(), parallaxFac=1.0):
        self.gBuffer = gBuffer
        self.parallaxFac = parallaxFac
        self.image = self.gBuffer.g0


def loadImage(path, parallaxFac=1.0):
    p = os.path.abspath(path)

    image = pg.image.load(p)
    g1 = f"{os.path.split(p)[0]}\\{os.path.split(p)[1].split('.')[0]}g1.png"
    if os.path.exists(g1):
        normal = pg.image.load(g1)
        normal.convert()
        if settings.debug: print(f"loaded normal map {g1}")
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
        self.size = self.animations.standing.getImage().gBuffer.g0.get_size()

        self.pos = [(GetSystemMetrics(0) / 2 / settings.scaleFactor) - (self.size[0] / 2),
                    (GetSystemMetrics(1) / 2 / settings.scaleFactor) - (self.size[0] / 2)]
        self.velocity = maths.Vec2((0, 0))
        self.rotation = rotation
        self.floor = None

        self.jumping = False
        self.falling = False
        self.inAir = True

        self.flipped = False
        self.current = self.animations.standing
        self.movementStart = 0
        self.oldOffset = -1
        self.walk = True

    def update(self, ticks, keys, floor: pg.mask.Mask, offset):
        # TODO: pBuffer
        self.animations.cTicks(ticks)

        if settings.physics:
            self.floor = floor

            oldY = self.pos[1]
            y = 0

            while self.floor.get_at((round(self.pos[0] - offset + self.size[0] / 2), round(y + self.size[0]) - 1)) == 0:
                y += 1

                if y == settings.unscaledSize[1]:
                    break

            y -= 1

            floorY = y
            y = (oldY - (self.velocity.y * ticks * settings.gravity))
            self.velocity.y -= settings.gravity * ticks

            if y >= floorY:
                y = floorY
                self.velocity.y = 0

            if self.velocity.x > 0:
                self.flipped = False
            elif self.velocity.x < 0:
                self.flipped = True

            if self.flipped:
                # check left (-2)

                yL = 0
                while self.floor.get_at(
                        (round(self.pos[0] - (offset + 1) + self.size[0] / 2), round(yL + self.size[0]) - 1)) == 0:
                    yL += 1
                    if yL == settings.unscaledSize[1]:
                        break

                yL -= 1
                self.walk = not (y - yL > settings.maxStepSizeY)

            else:
                # check right (+2)

                yR = 0
                while self.floor.get_at(
                        (round(self.pos[0] - (offset - 1) + self.size[0] / 2), round(yR + self.size[0]) - 1)) == 0:
                    yR += 1
                    if yR == settings.unscaledSize[1]:
                        break

                yR -= 1
                self.walk = not (y - yR > settings.maxStepSizeY)

            self.pos[1] = y
            self.inAir = not (y == floorY)

        if keys[keyboardSettings.left] == 1:
            self.movementStart += ticks

            if keys[keyboardSettings.run] == 1:

                self.velocity.x = -(settings.speed * settings.runFac * ticks * 100)
                self.current = self.animations.running
                if self.walk:
                    offset += settings.speed * settings.runFac * ticks * 100
            else:

                self.velocity.x = -(settings.speed * ticks * 100)
                self.current = self.animations.walking
                if self.walk:
                    offset += settings.speed * ticks * 100

        elif keys[keyboardSettings.right] == 1:
            self.movementStart += ticks

            if keys[keyboardSettings.run] == 1:

                self.velocity.x = settings.speed * settings.runFac * ticks * 100
                self.current = self.animations.running
                if self.walk:
                    offset -= settings.speed * settings.runFac * ticks * 100
            else:

                self.velocity.x = settings.speed * ticks * 100
                self.current = self.animations.walking
                if self.walk:
                    offset -= settings.speed * ticks * 100

        else:
            self.velocity.x = 0
            self.current = self.animations.standing
            self.movementStart = 0

        if keys[keyboardSettings.jump] == 1:
            if not self.jumping and not self.inAir:
                self.jumping = True
                self.velocity.y += settings.jumpHeight / settings.gravity
        else:
            self.jumping = False

        return offset

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
        mask = pg.mask.from_surface(self.current.getImage().gBuffer.g0, 1)
        pBuffer.blit(physics.PBuffer(
            p=mask.to_surface(setcolor=(128 + self.velocity.x, 128 + self.velocity.y, 0), unsetcolor=(128, 128, 0))),
            self.pos)


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

    limit = -sx + settings.unscaledSize[0]
    if offset < limit:
        offset = limit
    elif offset > 0:
        offset = 0

    gBuffer = src if isinstance(src, light.GBuffer) else src.gBuffer

    offsetX = round(offset * src.parallaxFac)
    win.blit(gBuffer, (offsetX, y))
    win.blit(gBuffer, (offsetX + round(sizeX / src.parallaxFac), y))

    return offset


# scene
class SceneLayout:

    def __init__(self, Images, Lights: light.LightHandler, Physics: physics.PhysicsHandler, cam: camera.Camera,
                 foliage: list,
                 floor: Image, nextSceneOffset: int,
                 _id: int):
        self.images = Images
        self.lights = Lights
        self.physic = Physics
        self.cam = cam
        self.foliage = foliage
        self.floor = pg.mask.from_surface(floor.gBuffer.g0, 128)
        self.nextSceneRect = nextSceneOffset
        self.id = _id

        self.canvasSize = settings.unscaledSize


class Scene:

    def __init__(self, layout: SceneLayout, character: Character, enabled=False):
        self.id = layout.id

        self.character: Character = character
        self.images: list = layout.images
        self.floor = layout.floor
        self.foliage = layout.foliage

        self.physicsHandler: physics.PhysicsHandler = layout.physic
        self.lightHandler: light.LightHandler = layout.lights
        self.cam = layout.cam

        self.canvasSize = layout.canvasSize
        self.imageSize = self.images[0].image.get_size()
        self.nextSceneRect = layout.nextSceneRect

        self.tick = 0
        self.lastStep = 0
        self.offset = 0
        self.__enabled = enabled

        self.win: light.GBuffer = light.GBuffer(self.canvasSize)
        self.pBuffer = physics.PBuffer(self.canvasSize)

        self.font = pg.font.SysFont("Arial", 14)

    def disable(self):
        self.__enabled = False

    def enable(self):
        self.__enabled = True

    @property
    def enabled(self):
        return self.__enabled

    def update(self, ticks, keys):
        if self.__enabled:
            if self.id in settings.disabledScenes:
                self.disable()

            else:
                self.tick = ticks

                if settings.character: self.offset = self.character.update(ticks, keys, self.floor, self.offset)
                if settings.physics: self.physicsHandler.update(ticks, self.pBuffer)

    def render(self):
        if self.__enabled:
            self.win.reset()

            for i in self.images:
                self.offset = renderWithOffset(self.win, i, self.offset, self.imageSize)

            if settings.foliage:
                fgBuffer = light.GBuffer()
                for f in self.foliage:
                    fgBuffer.reset()
                    fgBuffer.blit(f.render(), f.pos.pos)
                    self.offset = renderWithOffset(self.win, fgBuffer, self.offset,
                                                   self.imageSize)  # TODO: foliage prevents offset from changing - fix

            if settings.physics:
                self.offset = renderWithOffset(self.win, self.physicsHandler.evaluate(self.pBuffer), self.offset,
                                               self.imageSize)

            if settings.character:
                self.character.render(self.win)
                if self.offset < -self.nextSceneRect:
                    self.disable()
                    return pg.Surface((0, 0))

            if settings.light:
                self.lightHandler.preEvaluate(self.win)

            if settings.camera:
                self.cam.preRender()

            if any([settings.light, settings.camera]):
                self.lightHandler.shaderHandler.fragment()

            if settings.light:
                self.win.blit(self.lightHandler.evaluate(), (0, 0))

            if settings.character and not settings.characterAffectedByLight:
                self.character.render(self.win)

            out = pg.transform.scale(self.win.g0, (math.floor(self.win.g0.get_size()[0] * settings.scaleFactor),
                                                   settings.size[1]))

            if settings.showFps:
                if self.tick > 0:
                    out.blit(self.font.render(f"{int(1 / self.tick)}", False, (255, 255 if self.character.walk else 0, 255)), (0, 0))

            return out
        else:
            return pg.Surface((0, 0))


class MainScreenSceneLayout:

    def __init__(self, shaderHandler: shaders.shaderHandler, bgImage, cam: camera.Camera):
        self.cam = cam
        self.image = bgImage
        self.lights = light.LightHandler(shaderHandler, [light.pointLight(settings.center, colors.lightColors.cold, 1, 0)])

        self.canvasSize = settings.unscaledSize


class MainScreenScene:

    def __init__(self, layout: MainScreenSceneLayout, enabled=True):
        self.image = layout.image

        self.cam = layout.cam
        self.lightHandler: light.LightHandler = layout.lights

        self.canvasSize = layout.canvasSize
        self.imageSize = self.image.image.get_size()

        self.mx = round(pg.mouse.get_pos()[0] / settings.scaleFactor)
        self.my = round(pg.mouse.get_pos()[1] / settings.scaleFactor)

        self.win: light.GBuffer = light.GBuffer(settings.unscaledSize)
        self.font = pg.font.SysFont("Arial", 14)
        self.tick = 0
        self.pressed = False

        self.startButtonRect = pg.Rect((round(settings.center[0] / 2) - 20, round(settings.center[1] / 2) - 10),
                                       (40, 20))
        self.startButtonFont = pg.font.SysFont("Arial", 18)
        self.startButtonText = self.startButtonFont.render("Start", False, (0, 0, 0))

        self.__enabled = enabled

    def disable(self):
        self.__enabled = False

    def enable(self):
        self.__enabled = True

    @property
    def enabled(self):
        return self.__enabled

    def update(self, ticks, keys):
        if self.__enabled:
            self.mx = round(pg.mouse.get_pos()[0] / settings.scaleFactor)
            self.my = round(pg.mouse.get_pos()[1] / settings.scaleFactor)

            self.tick = ticks
            self.lightHandler.lights[0].x = self.mx
            self.lightHandler.lights[0].y = self.my

            self.lightHandler.lightsToList()

    def render(self):
        if self.__enabled:
            self.win.reset()

            collide = self.startButtonRect.collidepoint(self.mx, self.my)

            c = (255, 255, 255)

            if collide:

                if pg.mouse.get_pressed(3)[0]:  # switch scenes
                    self.disable()

                c = (176, 184, 220)  # BLUE: hovering

            self.win.blit(self.image.gBuffer, (0, 0))
            pg.draw.rect(self.win.g0, c, self.startButtonRect)
            self.win.g0.blit(self.startButtonText,
                             (self.startButtonRect.topleft[0] + 1, self.startButtonRect.topleft[1] - 1))

            if settings.light:
                self.lightHandler.preEvaluate(self.win)

            if settings.camera:
                self.cam.preRender()

            if any([settings.light, settings.camera]):
                self.lightHandler.shaderHandler.fragment()

            if settings.light:
                self.win.blit(self.lightHandler.evaluate(), (0, 0))

            if settings.camera: self.win.g0 = self.cam.render()
            out = pg.transform.scale(self.win.g0, (math.floor(self.win.g0.get_size()[0] * settings.scaleFactor),
                                                   math.floor(self.win.g0.get_size()[1] * settings.scaleFactor)))

            if settings.showFps:
                if self.tick > 0:
                    out.blit(self.font.render(f"{int(1 / self.tick)}", False, (255, 255, 255)), (0, 0))

            return out
        else:
            return pg.Surface((0, 0))
