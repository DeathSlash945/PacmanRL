import pygame as pg
import sys
import map as maze
from env import Game

HUMAN=True #kept for debugginf and for playing myself :P

pg.init()
panelW=340
height,width=224,288
s=pg.display.set_mode((height*maze.winScale+panelW,width*maze.winScale))
pg.display.set_caption("Pacman :P")

clock=pg.time.Clock()
font=pg.font.SysFont('consolas',15)

game=Game(fx=True)
ready_font=pg.font.SysFont('consolas',28,bold=True)
pacman,ghosts=game.pacman,game.ghosts #short names for drawing

keys={pg.K_LEFT:'left',pg.K_RIGHT:'right',pg.K_UP:'up',pg.K_DOWN:'down',
      pg.K_a:'left',pg.K_d:'right',pg.K_w:'up',pg.K_s:'down'} #both wasd and arrow keys work

t=0
def draw():
    s.fill(maze.bgColor)
    maze.build_walls(s,maze.walls)
    maze.draw_pellets(s,t)
    if game.fruit:
        x,y=maze.cellPos(13.5,17)
        pg.draw.circle(s,'red',(int(x),int(y)),int(0.7*maze.gridSize)) #placeholder
    if game.dying: #ghosts vanish, pac wedge opens until gone
        m=int(180*(1-game.freeze/90))
        if m<175: pacman.draw(s,t,m)
    else:
        pop=game.popup
        if not pop: pacman.draw(s,t) #pac hidden while the score shows
        for g in ghosts:
            if pop and g.mode=='eaten' and g.tile==pop[0]: continue
            g.draw(s,t)
        if pop:
            x,y=maze.cellPos(*pop[0])
            s.blit(font.render(str(pop[1]),False,'cyan'),(x-10,y-6))
        elif game.freeze>0:
            r=ready_font.render("READY!",False,'yellow').get_rect(center=maze.cellPos(13.5,17))
            s.blit(ready_font.render("READY!",False,'yellow'),r)
    draw_panel()
    pg.display.flip()

def draw_panel():
    x,y=224*maze.winScale+10,10
    for k,v in game.info().items():
        s.blit(font.render(f"{k}: {v}",False,'yellow'),(x,y)); y+=18
    y+=12

    for e in [pacman]+ghosts:
        for k,v in e.state().items():
            s.blit(font.render(f"{k}: {v}",False,'white'),(x,y)); y+=18
        y+=12

over_t=0
running=True
while running:
    for event in pg.event.get():
        if event.type==pg.QUIT: running=False
        if HUMAN and event.type==pg.KEYDOWN and event.key in keys:
            pacman.set_dir(keys[event.key])
    if game.game_over and HUMAN: #2s pause, then restart
        over_t+=1
        if over_t>120: over_t=0; game.reset()
    game.update()
    if game.freeze==0: t+=1
    draw()
    clock.tick(60)

pg.quit()
sys.exit()