import pygame as pg
import map as maze
from pacman import DIRS,OPP
import math,random

from levels import MAX

NOUP={(12,11),(15,11),(12,23),(15,23)}   #ghosts can't turn up here (not when frightened)

#modes: 'house','scatter','chase','frightened','eaten'
class Ghost:
    P=None
    elroy=0
    white=False
    base ='scatter'
    def __init__(self,name,color,start,scatter,mode='house',release=0):
        self.name=name
        self.color=color
        self.start=start
        self.scatter=scatter #scatter corner target
        self.init_mode=mode
        self.release=release
        self.reset()

    def reset(self):
        self.x,self.y=self.start
        self.mode=self.init_mode
        self.dir='left'
        self.target=self.scatter
        self.reverse=False
        self.path=[]

    def set_mode(self,m):
        if self.mode in ('scatter','chase'):
            self.mode=m; self.reverse=True

    def frighten(self):
        if self.mode in ('scatter','chase'):
            self.mode='frightened'; self.reverse=True

    def leave(self):
        self.mode='leaving'; self.path=[(13.5,self.y),(13.5,11)]

    def walk(self,d):                  #axis-aligned waypoint walk
        while d>1e-9 and self.path:
            tx,ty=self.path[0]
            dx,dy=tx-self.x,ty-self.y
            dist=abs(dx)+abs(dy)
            if dist<1e-9: self.x,self.y=tx,ty; self.path.pop(0); continue
            step=min(d,dist)
            if dx: self.x+=math.copysign(step,dx); self.dir='right' if dx>0 else 'left'
            else: self.y+=math.copysign(step,dy); self.dir='down' if dy>0 else 'up'
            d-=step

    def set_target(self,pacman,blinky):
        px,py=pacman.tile; dx,dy=DIRS[pacman.dir]
        if self.mode=='scatter': 
            self.target=(px,py) if (self.name=='blinky' and Ghost.elroy) else self.scatter
        elif self.mode=='chase':
            if self.name=='blinky': self.target=(px,py)
            elif self.name=='pinky': self.target=(px+4*dx-(4 if dy<0 else 0),py+4*dy) #original up-bug
            elif self.name=='inky':
                ax,ay=px+2*dx-(2 if dy<0 else 0),py+2*dy
                bx,by=blinky.tile
                self.target=(2*ax-bx,2*ay-by)
            elif self.name=='clyde':
                far=math.hypot(px-self.tile[0],py-self.tile[1])>8
                self.target=(px,py) if far else self.scatter
        elif self.mode=='eaten': self.target=(13,11) 

    def choose(self):
        c,r=int(self.x),int(self.y)
        opts=[]
        for d in ('up','left','down','right'):                 #tie-break order
            if d==OPP[self.dir]: continue
            if d=='up' and (c,r) in NOUP and self.mode!='frightened': continue
            nx,ny=c+DIRS[d][0],r+DIRS[d][1]
            if not maze.is_wall(nx,ny,door=self.mode=='eaten'): opts.append((d,nx,ny))
        if not opts: return OPP[self.dir]
        if self.mode=='frightened': return random.choice(opts)[0]
        tx,ty=self.target
        return min(opts,key=lambda o:(o[1]-tx)**2+(o[2]-ty)**2)[0]   #min keeps first on ties

    @property
    def tile(self): return (math.floor(self.x+0.5),math.floor(self.y+0.5))

    @property
    def speed(self):
        P=Ghost.P
        if self.mode=='house': return 0
        if self.mode in ('eaten','entering'): return 0.25
        if self.mode=='leaving': return 0.0625
        if self.mode=='frightened': return MAX*P['gh_fr']
        if self.y==14 and (self.x<6 or self.x>21): return MAX*P['gh_tun']
        if self.name=='blinky' and Ghost.elroy: return MAX*P['el_sp'][Ghost.elroy-1]
        return MAX*P['gh']

    def update(self,pacman,blinky):
        if self.mode=='house': return
        if self.mode in ('leaving','entering'):
            self.walk(self.speed)
            if not self.path:
                if self.mode=='leaving': self.mode=Ghost.base; self.dir='left'
                else: self.mode='leaving'; self.path=[(13.5,11)]    #revived, walk back out
            return
        self.set_target(pacman,blinky)
        d=self.speed
        while d>1e-9:
            if self.x%1==0 and self.y%1==0:
                if self.mode=='eaten' and (int(self.x),int(self.y)) in ((13,11),(14,11)):
                    self.mode='entering'; self.path=[(13.5,self.y),(13.5,14)]
                    return
                if self.reverse: self.dir=OPP[self.dir]; self.reverse=False
                else: self.dir=self.choose()
            dx,dy=DIRS[self.dir]
            a=self.x if dx else self.y
            nxt=math.floor(a)+1 if (dx or dy)>0 else math.ceil(a)-1
            step=min(d,abs(nxt-a))
            self.x+=dx*step; self.y+=dy*step
            if step==abs(nxt-a): self.x,self.y=(nxt,self.y) if dx else (self.x,nxt) #snap
            d-=step
            if self.x<-1: self.x+=29 #tunnel
            if self.x>28: self.x-=29

    def draw(self,s,t):
        x,y=map(int,maze.cellPos(self.x,self.y)); r=int(0.7*maze.gridSize)
        if self.mode not in ('eaten','entering'):
            c=('white' if Ghost.white else 'blue') if self.mode=='frightened' else self.color
            pg.draw.circle(s,c,(x,y),r)
            pg.draw.rect(s,c,(x-r,y,2*r,r))
            k=r//3; ph=(t//8)%2 #feet swap every 8 frames
            for i in range(3):
                bx=x-r+(2*i+ph)*k-4
                pg.draw.polygon(s,maze.bgColor,[(bx,y+r+1),(bx+k,y+r-k),(bx+2*k,y+r+1)])
        if self.mode!='frightened':
            dx,dy=DIRS[self.dir]
            for sg in (-1,1):
                ex=x+sg*r//3
                pg.draw.circle(s,'white',(ex,y-r//4),r//3)
                pg.draw.circle(s,'blue',(ex+dx*r//6,y-r//4+dy*r//6),r//6) #pupils look where it is going

        if self.mode in ('scatter','chase'):
            tx,ty=maze.cellPos(*self.target)
            pg.draw.rect(s,self.color,(int(tx)-4,int(ty)-4,8,8),2)

    def state(self):
        return dict(name=self.name,
                    pos=(round(self.x,2),round(self.y,2)),
                    tile=self.tile,
                    dir=self.dir,
                    mode=self.mode,target=self.target,
                    speed=self.speed)