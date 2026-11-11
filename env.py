import random
import map as maze
import levels
from pacman import Pacman
from ghost import Ghost

class Game:
    def __init__(self,fx=True):
        self.fx=fx
        self.total=len(maze.pellets) #244 dots+pellets, used for fruit timing
        self.pacman=Pacman()
        self.ghosts=[Ghost('blinky','red',(13.5,11),(25,-3),'scatter'),
                     Ghost('pinky','pink',(13.5,14),(2,-3),'house',0),
                     Ghost('inky','cyan',(11.5,14),(27,31),'house',30),
                     Ghost('clyde','orange',(15.5,14),(0,31),'house',60)]
        self.reset()

    def reset(self,seed=None): #rl episode reset
        if seed is not None: random.seed(seed)
        self.level=1
        self.fruit=None
        self.combo=0
        self.game_over=False
        self.pacman.reset(); maze.reset_pellets()
        self.dying=False
        self.popup=None
        self.freeze=0
        self.load_level(); self.reset_round()

    def finish_death(self):
        self.dying=False
        if self.pacman.lives<=0: self.game_over=True
        else: self.reset_round()

    def load_level(self):
        self.P=levels.params(self.level)
        Ghost.P=Pacman.P=self.P
        self.sched=self.P['sched']; self.phase=self.timer=0
        self.ghosts[2].release=self.P['inky']; self.ghosts[3].release=self.P['clyde']

    def next_level(self):
        self.level+=1; self.fruit=None
        maze.reset_pellets(); self.load_level(); self.reset_round()

    def reset_round(self):
        self.pacman.respawn()
        for g in self.ghosts: g.reset()
        self.phase=self.timer=self.fright=self.dots=self.idle=0
        Ghost.base='scatter'
        self.freeze=120 if self.fx else 0

    def tick_mode(self):
        self.timer+=1
        dur=self.sched[self.phase][0]
        if dur and self.timer>=dur:
            self.phase+=1; self.timer=0
            Ghost.base=self.sched[self.phase][1]
            for g in self.ghosts: g.set_mode(self.sched[self.phase][1])

    def release(self):
        w=next((g for g in self.ghosts if g.mode=='house'),None)
        if w and (self.dots>=w.release or self.idle>=self.P['idle']):
            w.leave(); self.dots=0; self.idle=0

    def collide(self):
        pac=self.pacman
        for g in self.ghosts:
            if g.tile!=pac.tile: continue
            if g.mode=='frightened':
                g.mode='eaten'; g.reverse=False
                pts=200*2**self.combo
                pac.score+=pts; self.combo+=1
                if self.fx: self.freeze=60; self.popup=(g.tile,pts)
            elif g.mode in ('scatter','chase'):
                pac.alive=False; pac.lives-=1
                if self.fx: self.freeze=90; self.dying=True
                else: self.finish_death()
                return

    def update(self): #one frame, no drawing in here
        if self.game_over: return
        if self.freeze>0:
            self.freeze-=1
            if self.freeze==0:
                self.popup=None
                if self.dying: self.finish_death()
            return
        pac,ghosts,P=self.pacman,self.ghosts,self.P
        pac.update()
        p=maze.pellets.pop(pac.tile,None)
        if p:
            self.dots+=1; self.idle=0
            pac.score+=10 if p=='.' else 50
            pac.pause=1 if p=='.' else 3 #slows pac down while eating dots like original
            if p=='o':
                self.combo=0
                for g in ghosts:
                    if P['fright']: g.frighten()
                    elif g.mode in ('scatter','chase'): g.reverse=True #fright time 0 still reverses
                self.fright=P['fright']*60
        else: self.idle+=1
        if self.fright>0: #scatter/chase timer paused meanwhile
            self.fright-=1
            if self.fright==0:
                for g in ghosts:
                    if g.mode=='frightened': g.mode=Ghost.base #no reversal on exit
        else: self.tick_mode()

        pac.fright=self.fright>0
        Ghost.white=0<self.fright<=P['flash']*14 and (self.fright//7)%2==0 #14 frames per flash
        rem=len(maze.pellets)
        Ghost.elroy=2 if rem<=P['elroy']//2 else 1 if rem<=P['elroy'] else 0

        eaten=self.total-rem
        if p and eaten in (70,170): self.fruit=[*P['fruit'],random.randint(560,620)] #~9.3 to 10.3 s
        if self.fruit:
            self.fruit[2]-=1
            if pac.tile in ((13,17),(14,17)): pac.score+=self.fruit[1]; self.fruit=None
            elif self.fruit[2]<=0: self.fruit=None

        self.release()
        for g in ghosts: g.update(pac,ghosts[0])
        self.collide()
        if not maze.pellets: self.next_level()

    def info(self): #same dict for the panel and for rl obs later
        return dict(level=self.level,fruit=self.fruit,fright=self.fright,
                    phase=self.sched[self.phase][1],dots_left=len(maze.pellets),
                    elroy=Ghost.elroy,game_over=self.game_over,freeze=self.freeze)