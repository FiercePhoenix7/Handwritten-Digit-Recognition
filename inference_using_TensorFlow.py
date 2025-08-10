import pygame
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model

####################### SETTINGS ###########################
pygame.init()

pixel_width = 28
pixel_height = 28
pixel_side = 20


WIDTH = pixel_width*pixel_side + 300
HEIGHT = pixel_height*pixel_side

BLACK = (0, 0, 0)
RED = (225, 0, 0)
GREEN = (0, 225, 0)
GREY = (107, 107, 107)
WHITE = (225, 225, 225)

FPS = 57

######################## APP CLASS ##########################

class App:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.mouse_pos = (0, 0)
        self.pressed = False
        self.image = np.zeros((pixel_height, pixel_width), dtype='int')
        self.model = load_model("Model.h5")
        self.prediction = None
        self.running = True

    def run(self):
        while self.running:
            self.events()
            self.draw()
        pygame.quit()
        sys.exit()
    
    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.pressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.pressed = False
                self.image = self.pixelize()
                for h in range(pixel_height):
                    for w in range(pixel_width):
                        if self.image[h][w] != 0:
                            pygame.draw.rect(self.screen, (self.image[h][w]*255,)*3, (w*pixel_side + 2, h*pixel_side + 2, pixel_side - 1, pixel_side - 1))
                self.prediction = tf.nn.softmax(self.model.predict(self.image.reshape(-1, 28 * 28).astype("float32"))).numpy()
                self.display_prediction()
            if self.pressed:
                self.mouse_pos = pygame.mouse.get_pos()

    def draw(self):
        if self.pressed:   
            pygame.draw.circle(self.screen, WHITE, self.mouse_pos, 20)
        
        elif self.prediction is not None:
            self.screen.fill(BLACK)
            self.display_prediction()
            self.draw_grid(pixel_width*pixel_side, pixel_height*pixel_side)
            for h in range(pixel_height):
                for w in range(pixel_width):
                    if self.image[h][w] != 0:
                        pygame.draw.rect(self.screen, (self.image[h][w]*255,)*3, (w*pixel_side + 1, h*pixel_side + 1, pixel_side - 1, pixel_side - 1))
            pygame.display.update()
        else:
            self.screen.fill(BLACK)
            self.draw_grid(pixel_width*pixel_side, pixel_height*pixel_side)
            for h in range(pixel_height):
                for w in range(pixel_width):
                    if self.image[h][w] != 0:
                        pygame.draw.rect(self.screen, (self.image[h][w]*255,)*3, (w*pixel_side + 1, h*pixel_side + 1, pixel_side - 1, pixel_side - 1))
        pygame.display.update()

###################### HELPER FUNCTIONS ########################

    def draw_grid(self, width, height):
        for i in range(0, width, pixel_side):
            pygame.draw.line(self.screen, GREY, (i, 0), (i, height))
        for i in range(0, height, pixel_side):
            pygame.draw.line(self.screen, GREY, (0, i), (width, i))
        pygame.draw.line(self.screen, GREY, (width, 0), (width, height))
        pygame.draw.line(self.screen, GREY, (0, height), (width, height))


    def pixelize(self):
        image = np.zeros((pixel_height, pixel_width), dtype='float')
        for h in range(pixel_height):
            for w in range(pixel_width):
                sum = 0
                for y in range(h*pixel_side + 1, (h + 1)*pixel_side):
                    for x in range(w*pixel_side + 1, (w + 1)*pixel_side):
                        sum += self.screen.get_at((x, y))[0]
                sum = sum/pixel_side**2
                image[h][w] = sum/255
        return image
                
    def display_prediction(self):
        font = pygame.font.Font('freesansbold.ttf', 27)
        text = font.render("Digit    Confidence", True, WHITE)
        textRect = text.get_rect()
        textRect.topleft = (WIDTH - 275, 35)
        self.screen.blit(text, textRect)
        for digit in range(len(self.prediction[0])):
            if digit == np.argmax(self.prediction):
                text = font.render(str(digit) + f'    :    {"{:.2f}".format(self.prediction[0][digit]*100)}%', True, GREEN)
            else:
                text = font.render(str(digit) + f'    :    {"{:.2f}".format(self.prediction[0][digit]*100)}%', True, WHITE)
            textRect = text.get_rect()
            textRect.topleft = (WIDTH - 250, 80 + digit*45)
            self.screen.blit(text, textRect)

###############################################################

app = App()
app.run()
