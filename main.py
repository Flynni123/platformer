import math

import pygame as pg
import matplotlib.pyplot as pp
import time as t

import colors
import physics
import light
import scene
import settings

"""
Instead of rendering the object and passing it to the Display class, i'm passing the object to render to the Display
class and then Rendering it. if i want to change it, i just change it.
- Tzu Sun, art of not knowing what i'm programming
"""


class pixelatingMethods:
    lame = 0
    fancy = 1


class Display:

    def __init__(self, size=(0, 0), fullscreen=True, time=scene.TimeHandler()):
        """
        :param size: window size
        :param fullscreen: if window should be in fullscreen. ignores size argument
        :param time: handles time
        """

        self.size = size
        self.fullscreen = fullscreen
        self.timeHandler = time

        # init EVERYTHING
        pg.init()
        pg.display.init()
        pg.mixer.init()

        self.clock = pg.time.Clock()

        self.fpsList = []
        self.timeList = []

        if fullscreen:
            self.disp = pg.display.set_mode(size, pg.FULLSCREEN)
        else:
            self.disp = pg.display.set_mode(size)

        self.s = scene.Scene(scene.SceneLayout([
            scene.Image("assets/images/bg2.png", 0.4),
            scene.Image("assets/images/bg1.png", 1),
            scene.Image("assets/images/bg0.png", 1)
        ], light.LightHandler([
            light.pointLight((10, 10), colors.lightColors.cold, 1, .1, 1.2)
        ]), physics.PhysicsHandler([
            physics.line((90, 10))
        ])),
            scene.Character(
                scene.Animation("assets/images/character.png", 1)
            ), True)

        self.run = False
        firstRun = True

        self.run = True
        while self.run:

            if settings.debug:
                self.s.lightHandler.lights[0].x = pg.mouse.get_pos()[0] / settings.scaleFactor
                self.s.lightHandler.lights[0].y = pg.mouse.get_pos()[1] / settings.scaleFactor

            if firstRun:
                firstRun = False
                self.timeHandler.start()
                self.disp.fill(colors.black)

            ticks = self.timeHandler.getTicks(self.clock.get_fps())
            self.s.update(ticks, pg.key.get_pressed())

            for event in pg.event.get():

                if event.type == pg.QUIT:

                    self.run = False

                elif event.type == pg.KEYDOWN:

                    if event.key == pg.K_ESCAPE:
                        self.run = False

            self.disp.blit(self.s.render(), (0, 0))
            pg.display.update()
            self.clock.tick()

            if not self.run:
                print(round(self.clock.get_fps(), 1))

        pg.quit()


d = Display()

if settings.debug:
    pp.plot(d.timeHandler.tickList)
    pp.plot(d.timeHandler.smoothList)
    pp.legend(["error", "tickValue", "smoothed", "predicted"])
    pp.show()
