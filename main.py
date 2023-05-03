import sys

import pygame as pg

import shaders

pg.init()

import camera
import colors
import maths
import physics
import light
import scene
import settings

"""
Instead of rendering everything as efficiently as possible it feel's like I'm rendering everything twice.
Oh, and memory management is hell.
- Tzu Sun, art of not knowing what i'm coding
"""


class Display:

    def __init__(self, size=(0, 0), fullscreen=settings.fullscreen, time=scene.TimeHandler()):
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

        self.display = pg.display.set_mode(self.size, pg.FULLSCREEN) if fullscreen else pg.display.set_mode(self.size)

        shaderHandler = shaders.shaderHandler(settings.unscaledSize)
        cam = camera.Camera(shaderHandler)

        character = scene.Character(
            scene.AnimationHandler({
                "walking": scene.Animation(scene.loadImage("assets/images/character_walking.png"), .2),
                "running": scene.Animation(scene.loadImage("assets/images/character_running.png"), .2),
                "standing": scene.Animation(scene.loadImage("assets/images/character_standing.png"), .2),
                "sitting": scene.Animation(scene.loadImage("assets/images/character_standing.png"), .2)
            })
        )
        self.scenes = [
            # MAIN SCREEN SCENE
            scene.MainScreenScene(scene.MainScreenSceneLayout(shaderHandler, scene.loadImage("assets/images/MainScreen/bg.png"), cam)),

            # FIRST SCENE
            scene.Scene(
                scene.SceneLayout([
                    scene.loadImage("assets/images/scene1/bg2.png", 0.4),
                    scene.loadImage("assets/images/scene1/bg1.png", 1),
                    scene.loadImage("assets/images/scene1/bg0.png", 1)
                    ],
                    light.LightHandler(shaderHandler, [
                        light.pointLight((10, 10), colors.lightColors.cold, 1, .05, 1.2)
                    ]),
                    physics.PhysicsHandler(shaderHandler, [

                    ]),
                    cam,
                    foliage=[],
                    floor=scene.loadImage("assets/images/scene1/bg1.png", 1),
                    nextSceneOffset=780,
                    _id=0
                ),
                character=character
            ),

            # SECOND SCENE
            scene.Scene(
                scene.SceneLayout([
                    scene.loadImage("assets/images/scene2/bg2.png", 1/3),
                    scene.loadImage("assets/images/scene2/bg1.png", 2/3),
                    scene.loadImage("assets/images/scene2/bg0.png", 3/3)
                    ],
                    light.LightHandler(shaderHandler, [
                        light.pointLight(xy, colors.lightColors.cold, .3, .05, 1) for xy in [(30, 50), (162, 50)]
                    ]),
                    physics.PhysicsHandler(shaderHandler, [

                    ]),
                    cam,
                    foliage=[],
                    floor=scene.loadImage("assets/images/scene2/floor.png"),
                    nextSceneOffset=1000,
                    _id=1
                ),
                character=character
            )
        ]

        self.testScenes = [
            scene.Scene(
                scene.SceneLayout([
                    scene.loadImage("assets/images/testScene/bg.png")
                ],
                    light.LightHandler(shaderHandler, [
                        light.spotLight((10, 10), colors.lightColors.cold, 1, 50, 20, .5),
                        light.globalLight(intensity=.2)
                    ]),
                    physics.PhysicsHandler(shaderHandler, [

                    ]),
                    cam,
                    foliage=[],
                    floor=scene.loadImage("assets/images/testScene/bg.png"),
                    nextSceneOffset=1000,
                    _id=0
                ),
                character=character
            )
        ]

        self.counter = 0
        self.currentList = self.testScenes
        self.current = self.currentList[self.counter]

        for e, s in enumerate(self.currentList):
            if isinstance(s, scene.Scene) or isinstance(s, scene.MainScreenScene):
                if e == 0:
                    s.enable()
                else:
                    s.disable()

        pg.mouse.set_visible(False)

        self.run = True

        self.timeHandler.start()
        self.display.fill(colors.black)

        while self.run:

            if settings.debug:
                self.current.lightHandler.lights[0].x = pg.mouse.get_pos()[0] / settings.scaleFactor
                self.current.lightHandler.lights[0].y = pg.mouse.get_pos()[1] / settings.scaleFactor

            ticks = self.timeHandler.getTicks()
            self.current.update(ticks, pg.key.get_pressed())

            if not self.current.enabled:
                self.counter += 1
                self.current = self.currentList[self.counter]
                self.current.enable()
                self.current.update(ticks, pg.key.get_pressed())

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.run = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self.run = False

            self.display.blit(self.current.render(), (0, 0))
            pg.display.update()
            self.clock.tick()

        print(round(self.clock.get_fps(), 1))
        pg.mouse.set_visible(True)
        pg.quit()
        sys.exit()


d = Display()
