import pygame as pg
import map as maze
import math

from levels import MAX

pacmanSize=0.6
DIRS={'left':(-1,0),'right':(1,0),'up':(0,-1),'down':(0,1),'none':(0,0)}
OPP={'left':'right','right':'left','up':'down','down':'up','none':'none'}

class Pacman:
    P=None
    def __init__(self):
        self.reset()

    def reset(self):
        self.moving=True
        self.x,self.y=13.5,23 #tile units, tile centers are integers
        self.dir='left'
        self.next_dir='left' #buffered turn
        self.pause=0
        self.fright=False
        self.alive=True
        self.lives=3
        self.score=0
        self.color='yellow'

    def respawn(self):
        self.x,self.y=13.5,23
        self.dir=self.next_dir='left'
        self.alive=True
        self.pause=0
        self.fright=False
    
    @property
    def tile(self): return (math.floor(self.x+0.5),math.floor(self.y+0.5))

    @property
    def speed(self):      #80% of max normally, 90% during fright
        return MAX*(Pacman.P['pac_fr'] if self.fright else Pacman.P['pac'])

    def set_dir(self,d): self.next_dir=d

    def update(self):
        if self.pause>0: self.pause-=1; return
        if self.next_dir==OPP[self.dir]: self.dir=self.next_dir; self.moving=True   #instant reverse
        d=self.speed
        while d>1e-9:
            if self.x%1==0 and self.y%1==0: #on a tile centre
                c,r=int(self.x),int(self.y)
                nx,ny=DIRS[self.next_dir]
                if not maze.is_wall(c+nx,r+ny): self.dir=self.next_dir
                dx,dy=DIRS[self.dir]
                self.moving=not maze.is_wall(c+dx,r+dy)
                if not self.moving: break
            dx,dy=DIRS[self.dir]
            a=self.x if dx else self.y
            nxt=math.floor(a)+1 if (dx or dy)>0 else math.ceil(a)-1
            step=min(d,abs(nxt-a))
            self.x+=dx*step; self.y+=dy*step
            if step==abs(nxt-a): self.x,self.y=(nxt,self.y) if dx else (self.x,nxt) #snap
            d-=step
            #for tunnel wapr
            if self.x<-1: self.x+=29
            if self.x>28: self.x-=29

    def draw(self,s,t,m=None):
        x,y=maze.cellPos(self.x,self.y)
        r=int(pacmanSize*maze.gridSize)
        face={'right':0,'down':90,'left':180,'up':270,'none':0}[self.dir]
        if m is None: m=[0,20,50,20][(t//2)%4] if self.moving else 20
        pts=[(x,y)]
        for a in range(m,360-m+1,5):
            rad=math.radians(a+face)
            pts.append((x+r*math.cos(rad),y+r*math.sin(rad)))
        pg.draw.polygon(s,self.color,pts)

    def state(self):
        return dict(name='pacman',
                    pos=(round(self.x,2),round(self.y,2)),
                    tile=self.tile,
                    dir=self.dir,next=self.next_dir,
                    speed=self.speed,
                    lives=self.lives,
                    score=self.score,alive=self.alive,
                    moving=self.moving)