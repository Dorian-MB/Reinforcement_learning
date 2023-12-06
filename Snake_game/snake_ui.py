import random
import numpy as np
import pygame 
from constant import *


def init():
    pygame.init()
    pygame.font.init()
    dis = pygame.display.set_mode(DISPLAY_RES, pygame.RESIZABLE)
    font = pygame.font.SysFont("verdana", FONT_SIZE)
    pygame.display.update()
    pygame.display.set_caption('Snake game')
    return dis, font
        
def init_board(dis):
    dis.fill(WHITE)
    background = pygame.Rect(*GAME_SIZE)
    pygame.draw.rect(dis, GRAY, background)
    border_dis = pygame.Rect(*BORDER_SIZE)
    pygame.draw.rect(dis, BLACK, border_dis, THICKNESS)

def draw_score(dis, font, score):
    text = font.render(f'Score: {score}', True, BLACK)
    score_coo = (SCORE_COO[0] , SCORE_COO[1]-text.get_height()//2)
    dis.blit(text, score_coo)

class Snake_cell(pygame.Rect):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        
    def collidefood(self, food):
        return self.colliderect(food.hit_box)
    
    def collidebody(self, boddy):
        return self.colliderect(boddy)

class Snake(Snake_cell):
    def __init__(self, x, y, width, height):
        head = Snake_cell(x, y, width, height)
        self.current_direction = None
        self.body = [[head, self.current_direction]]
             
    @property
    def head(self):
        return self.body[0][0]
    
    def draw(self, dis):
        for snake_cell, _ in self.body:
            pygame.draw.rect(dis, BLACK, snake_cell)
        
    def check_collision(self):
        for snake_cell, _ in self.body[1:]:
            if self.head.collidebody(snake_cell):
                self.current_direction = None  
                return False  
        return True
    
    def check_border_collision(self, border):
        if not self.head.colliderect(border):
            self.current_direction = None
            return False
        return True
    def move(self):
        for snake_cell, direction in self.body:
            snake_cell.move_ip(*direction)
            
        for i in range(len(self.body)-1, 0, -1):
            self.body[i][1] = self.body[i-1][1]
    
    def grow(self, former_tail_x, former_tail_y, tail_direction):
        snake_tail = Snake_cell(former_tail_x, former_tail_y, SNAKE_SIZE, SNAKE_SIZE)
        self.body.append([snake_tail, tail_direction])

class Food:
    def __init__(self):
        self.x = random.randint(0, N-1) # include
        self.y = random.randint(0, N-1)
        self.coordinate = X+(self.x + 1/2) * SNAKE_SIZE, Y+(self.y + 1/2) * SNAKE_SIZE
        self.radius = RADIUS
        
        self.hit_box = pygame.Rect(X+self.x*SNAKE_SIZE, Y+self.y*SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE)
    
    def draw(self, dis):
        pygame.draw.circle(dis, BLUE, self.coordinate, self.radius)
        
    def new_food(self):
        self.__init__()

class SnakeGame:
    def __init__(self):
        self.border = pygame.Rect(*GAME_SIZE)
        self.snake = Snake(*INIT_SNAKE_COO)
        self.food = Food()
        self.score = 0
        self.pause = False
        self.snake_grow = False
        self.game_over = False
        self.is_render_mode = False
        self.raw_obs = self.get_raw_observation()
        
    def init_pygame(self):
        self.dis, self.font = init()  
        self.clock = pygame.time.Clock()

        
    def play(self):

        self.init_pygame()
        while not self.game_over:
            init_board(self.dis)
            self.get_input_user()
            self.step(not_playing=False, draw=True)
            self.clock.tick(SNAKE_SPEED + 1/2*self.score)

    def get_input_user(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                
            if event.type == pygame.KEYDOWN :
                if event.key == pygame.K_RETURN:
                    self.snake = Snake(*INIT_SNAKE_COO)
                    self.food.new_food()
                    self.score = 0

                if event.key == pygame.K_SPACE:
                    if not self.pause :
                        self.pause = True
                        self.old_direction = self.snake.current_direction
                        self.snake.current_direction = None
                    else :
                        self.pause = False
                        self.snake.current_direction = self.old_direction
                        
                if event.key in DIRECTION: 
                    self.snake.current_direction = DIRECTION[event.key]
                    self.snake.body[0][1] = self.snake.current_direction
                
    def step(self, action=None, not_playing=True, draw=False) :
        if action is not None:
            self.snake.current_direction = list(DIRECTION.values())[action]
            self.snake.body[0][1] = self.snake.current_direction
        
        tail, tail_direction = self.snake.body[-1]
        tail_x, tail_y = tail.x, tail.y
        
        if self.snake.current_direction is not None:
            self.snake.move()

        border_collision = self.snake.check_border_collision(self.border)
        snake_collision = self.snake.check_collision()
        
        if self.snake.head.collidefood(self.food):
            self.food.new_food()
            self.score += 1
            self.snake_grow = True
            
        if draw and (snake_collision and border_collision) :
            self.draw()
        
        if self.snake_grow : # will be draw next iteration
            self.snake.grow(tail_x, tail_y, tail_direction)
            self.snake_grow = False 
        
        if not_playing:
            done = False if snake_collision and border_collision else True
            if not done :
                self.raw_obs = self.get_raw_observation()
            return self.raw_obs, self.score, done, {}
    
    def get_raw_observation(self):
        raw_obs = np.zeros((N, N), dtype=np.int32)
        for cell, _ in self.snake.body:
            x, y = (cell.x-X)//SNAKE_SIZE, (cell.y-Y)//SNAKE_SIZE
            raw_obs[y, x] = 1
        raw_obs[self.food.y, self.food.x] = 2
        return raw_obs
    
    def draw(self):
        if not self.is_render_mode:  
            self.init_pygame()
            self.is_render_mode = True
            
        init_board(self.dis)  
        self.snake.draw(self.dis)
        self.food.draw(self.dis)
        draw_score(self.dis, self.font, self.score)
        pygame.display.flip()
    
    def quit(self):
        pygame.quit() 
        
        
def test_speed():
    from time import time
    game = SnakeGame()
    action = 1
    i = 0
    T = time()
    for i in range(100_000):
        i+=1
        action = action%2+1
        game.step(action-1) # up/down
    return time()-T
    
if __name__ == "__main__":
    print("\ntest_speed: ", test_speed(), "\n")

    game = SnakeGame()
    game.play()
    game.quit()