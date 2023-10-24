import asyncio
import sys
import cv2
import pygame
from src.flappy import Flappy

if __name__ == "__main__":
    flappy = Flappy()
    try:
        asyncio.run(flappy.start())
    finally:
        cv2.destroyAllWindows()  # Certifique-se de fechar a janela da câmera
        pygame.quit()
        sys.exit()
