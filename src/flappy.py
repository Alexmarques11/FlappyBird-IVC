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

    async def start(self):
        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            self.previous_frame = None  # Inicialize o atributo previous_frame

            # Iniciar a captura da câmera em uma thread separada
            camera_thread = threading.Thread(target=self.capture_camera)
            camera_thread.daemon = True
            camera_thread.start()

            await self.splash()
            await self.play()
            await self.game_over()

    def ivc_rgb_to_hsv(self, src):
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        return hsv

    def capture_camera(self):
        while True:
            ret, image = self.cap.read()
            if not ret:
                break

            image_hsv = self.ivc_rgb_to_hsv(image)

            # Modifique o componente Hue (canal 0) para 50
            image_hsv[:, :, 0] = 50

            # Converta de volta para BGR (opcional, dependendo do que você deseja exibir)
            image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

            # Variável current_gray fora da estrutura condicional
            current_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            # Detecção de movimento
            if self.previous_frame is not None:
                # Calcule a diferença entre os quadros
                frame_diff = cv2.absdiff(self.previous_frame, current_gray)

                # Aplique um limiar para destacar as áreas de movimento
                _, thresholded = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

                # Realize operações de pós-processamento na imagem thresholded para detecção de movimento
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Defina um limite de área para detecção de movimento
                        # Atualize a posição do jogador ou realize outras ações do jogo com base no movimento
                        pass

                self.previous_frame = current_gray

            cv2.imshow("Image", image_bgr)

            c = cv2.waitKey(1)
            if c == 27:
                break
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

    def __del__(self):
        self.cap.release()  # Libere a câmera quando o objeto Flappy é destruído