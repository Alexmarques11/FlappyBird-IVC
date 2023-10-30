import asyncio
import sys
import cv2
import pygame
import threading
import numpy as np
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window

class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )
        self.cap = cv2.VideoCapture(0)  # Inicialize a captura de vídeo aqui

        # Variáveis de segmentação
        self.hmax = 180
        self.hmin = 0
        self.smax = 255
        self.smin = 0
        self.vmax = 255
        self.vmin = 0

        # Variáveis para exibir janelas
        self.show_camera = True
        self.show_mask = False
        self.show_mask_filtered = False

    async def start(self):
        # Crie uma thread separada para a segmentação de imagem
        threading.Thread(target=self.segmentation_thread, daemon=True).start()

        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            self.previous_frame = None  # Inicialize o atributo previous_frame

            await self.splash()
            await self.play()
            await self.game_over()

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
            event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)

    def update_segmentation(self, image_hsv):
        if self.hmin < self.hmax:
            ret, mask_hmin = cv2.threshold(src=image_hsv[:, :, 0], thresh=self.hmin, maxval=1, type=cv2.THRESH_BINARY)
            ret, mask_hmax = cv2.threshold(src=image_hsv[:, :, 0], thresh=self.hmax, maxval=1,
                                           type=cv2.THRESH_BINARY_INV)
            mask_h = mask_hmin * mask_hmax
        else:
            ret, mask_hmin = cv2.threshold(src=image_hsv[:, :, 0], thresh=self.hmin, maxval=1, type=cv2.THRESH_BINARY)
            ret, mask_hmax = cv2.threshold(src=image_hsv[:, :, 0], thresh=self.hmax, maxval=1,
                                           type=cv2.THRESH_BINARY_INV)
            mask_h = cv2.bitwise_or(mask_hmin, mask_hmax)

        ret, mask_smin = cv2.threshold(src=image_hsv[:, :, 1], thresh=self.smin, maxval=1, type=cv2.THRESH_BINARY)
        ret, mask_smax = cv2.threshold(src=image_hsv[:, :, 1], thresh=self.smax, maxval=1, type=cv2.THRESH_BINARY_INV)
        ret, mask_vmin = cv2.threshold(src=image_hsv[:, :, 2], thresh=self.vmin, maxval=1, type=cv2.THRESH_BINARY)
        ret, mask_vmax = cv2.threshold(src=image_hsv[:, :, 2], thresh=self.vmax, maxval=1, type=cv2.THRESH_BINARY_INV)

        mask_s = mask_smin * mask_smax
        mask_v = mask_vmin * mask_vmax
        mask = mask_h * mask_s * mask_v

        # Pré-processamento da imagem em escala de cinza
        grayscale_image = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2GRAY)


        equ = cv2.equalizeHist(grayscale_image)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(equ, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        mask_filtered = np.zeros(mask.shape, np.uint8)
        for i in range(len(contours)):
            contour = contours[i]
            contour_area = cv2.contourArea(contour)
            if contour_area > 100:
                M = cv2.moments(contour)
                Cx = int(np.round(M['m10'] / M['m00']))
                Cy = int(np.round(M['m01'] / M['m00']))
                perimeter = cv2.arcLength(curve=contour, closed=True)
                cv2.drawContours(image=mask_filtered, contours=[contour], contourIdx=-1, color=(255, 255, 255),
                                 thickness=cv2.FILLED)
                # Atualize a posição do jogador com base em Cy
                self.player.update_position(Cy)

        cv2.imshow("Mask Filtered", mask_filtered)
        return mask, mask_filtered
    def segmentation_thread(self):
        cv2.namedWindow("Camera")
        cv2.namedWindow("Segmentation Trackbars")
        cv2.createTrackbar("Hmin", "Segmentation Trackbars", self.hmin, 180, self.on_change_hmin)
        cv2.createTrackbar("Hmax", "Segmentation Trackbars", self.hmax, 180, self.on_change_hmax)
        cv2.createTrackbar("Smin", "Segmentation Trackbars", self.smin, 255, self.on_change_smin)
        cv2.createTrackbar("Smax", "Segmentation Trackbars", self.smax, 255, self.on_change_smax)
        cv2.createTrackbar("Vmin", "Segmentation Trackbars", self.vmin, 255, self.on_change_vmin)
        cv2.createTrackbar("Vmax", "Segmentation Trackbars", self.vmax, 255, self.on_change_vmax)

        while True:
            if not self.cap.isOpened():
                self.cap.open(0)
            ret, image = self.cap.read()
            image = image[:, ::-1, :]
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Aplicar a segmentação de imagem e detecção de objeto
            mask, mask_filtered = self.update_segmentation(image_hsv)

            # Exibir as imagens em janelas separadas
            if self.show_camera:
                cv2.imshow("Camera", image)
            if self.show_mask:
                cv2.imshow("Mask", mask)
            if self.show_mask_filtered:
                cv2.imshow("Mask Filtered", mask_filtered)

            c = cv2.waitKey(1)
            if c == 27:
                break

    def on_change_hmin(self, val):
        self.hmin = val

    def on_change_hmax(self, val):
        self.hmax = val

    def on_change_smin(self, val):
        self.smin = val

    def on_change_smax(self, val):
        self.smax = val

    def on_change_vmin(self, val):
        self.vmin = val

    def on_change_vmax(self, val):
        self.vmax = val

    def check_quit_event(self, event):
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    def __del__(self):
        self.cap.release()  # Libere a câmera quando o objeto Flappy é destruído