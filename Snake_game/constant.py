from pygame.constants import K_UP, K_DOWN, K_RIGHT, K_LEFT

FONT_SIZE = 36

DIS_X = 800
DIS_Y = 800
DISPLAY_RES = (DIS_X, DIS_Y)

X = 100
Y = 80
WIDTH = 600
HEIGHT = 600
THICKNESS = 6

GAME_SIZE = (X, Y, WIDTH, HEIGHT)
BORDER_SIZE = (X-THICKNESS, Y-THICKNESS, WIDTH+2*THICKNESS, HEIGHT+2*THICKNESS)

SNAKE_SIZE = 20
SNAKE_SPEED = 5
INIT_SNAKE_COO = (DIS_X/2, DIS_Y/2, SNAKE_SIZE, SNAKE_SIZE)
RADIUS = SNAKE_SIZE//2

SCORE_COO = (DIS_X//2-X, Y//2)
N = WIDTH//SNAKE_SIZE

BLUE = (89, 152, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
WHITE = (245, 245, 245)

STEP = SNAKE_SIZE 
DIRECTION = {
    K_UP : (0, -STEP),
    K_DOWN : (0, STEP),
    K_RIGHT : (STEP, 0),
    K_LEFT : (-STEP, 0)
}


