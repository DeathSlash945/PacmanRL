import pygame as pg
import math
import sys

#grid vars
winScale=4
gridSize=8*winScale
offset_y=24*winScale
offset_x=0
t=2*winScale
r=2*winScale
edgeColor='blue'
bgColor='black'

def cellPos(c,r):
    return (offset_x+c*gridSize+gridSize//2,offset_y+r*gridSize+gridSize//2)

def sign(v): return (v>0)-(v<0)
def unit(a,b): return (sign(b[0]-a[0]),sign(b[1]-a[1]))

def corner(p,i,closed):
    n=len(p)
    if not closed and i in (0,n-1): return None
    din=unit(p[i-1],p[i]); dout=unit(p[i],p[(i+1)%n])
    return None if din==dout else (din,dout)

def build_walls(s,walls):
    for pts in walls:
        p=[cellPos(*q) for q in pts]
        closed=p[0]==p[-1]
        if closed: p=p[:-1]
        n=len(p)
        cs=[corner(p,i,closed) for i in range(n)]
        for i in range(n if closed else n-1):#lines with no corners
            j=(i+1)%n
            d=unit(p[i],p[j])
            ka=r*(cs[i] is not None); kb=r*(cs[j] is not None)
            pg.draw.line(s,edgeColor,(p[i][0]+d[0]*ka,p[i][1]+d[1]*ka),(p[j][0]-d[0]*kb,p[j][1]-d[1]*kb),t)
        for i,c in enumerate(cs):#corners
            if c:
                din,dout=c
                cx=p[i][0]-din[0]*r+dout[0]*r
                cy=p[i][1]-din[1]*r+dout[1]*r
                sx=din[0]-dout[0]; sy=din[1]-dout[1]
                pg.draw.circle(s,edgeColor,(cx,cy),r,t,
                    draw_top_left=sx<0 and sy<0,draw_top_right=sx>0 and sy<0,
                    draw_bottom_left=sx<0 and sy>0,draw_bottom_right=sx>0 and sy>0)

def mirror_h(w):
    return [(27-c,r) for c,r in w]

floaties=[[(0,13),(5,13),(5,9),(0,9),(0,0),(13,0),(13,4),(14,4)],   #top border + tunnel
          [(0,15),(5,15),(5,19),(0,19),(0,30),(14,30)],             #bottom border + tunnel
          [(2,2),(5,2),(5,4),(2,4),(2,2)],
          [(7,2),(11,2),(11,4),(7,4),(7,2)],
          [(2,6),(5,6),(5,7),(2,7),(2,6)],
          [(7,6),(8,6),(8,9),(11,9),(11,10),(8,10),(8,13),(7,13),(7,6)],  #upper t
          [(7,15),(8,15),(8,19),(7,19),(7,15)],
          [(2,21),(5,21),(5,25),(4,25),(4,22),(2,22),(2,21)],
          [(7,21),(11,21),(11,22),(7,22),(7,21)],
          [(0,24),(2,24),(2,25),(0,25)],                            #border notch
          [(7,24),(8,24),(8,27),(11,27),(11,28),(2,28),(2,27),(7,27),(7,24)]]  #lower t

center=[[(10,6),(17,6),(17,7),(14,7),(14,10),(13,10),(13,7),(10,7),(10,6)],   #top t
        [(12,12),(10,12),(10,16),(17,16),(17,12),(15,12)],                    #ghost house, door gap
        [(10,18),(17,18),(17,19),(14,19),(14,22),(13,22),(13,19),(10,19),(10,18)],
        [(10,24),(17,24),(17,25),(14,25),(14,28),(13,28),(13,25),(10,25),(10,24)]]

walls=floaties+[mirror_h(w) for w in floaties]+center

MAZE=["############################",
      "#............##............#",
      "#.####.#####.##.#####.####.#",
      "#o####.#####.##.#####.####o#",
      "#.####.#####.##.#####.####.#",
      "#..........................#",
      "#.####.##.########.##.####.#",
      "#.####.##.########.##.####.#",
      "#......##....##....##......#",
      "######.##### ## #####.######",
      "######.##### ## #####.######",
      "######.##          ##.######",
      "######.## ###--### ##.######",
      "######.## #      # ##.######",
      "      .   #      #   .      ",
      "######.## #      # ##.######",
      "######.## ######## ##.######",
      "######.##          ##.######",
      "######.## ######## ##.######",
      "######.## ######## ##.######",
      "#............##............#",
      "#.####.#####.##.#####.####.#",
      "#.####.#####.##.#####.####.#",
      "#o..##.......  .......##..o#",
      "###.##.##.########.##.##.###",
      "###.##.##.########.##.##.###",
      "#......##....##....##......#",
      "#.##########.##.##########.#",
      "#.##########.##.##########.#",
      "#..........................#",
      "############################"]   # '#'wall, '-'ghost door, '.'dot, 'o'power

pellets={}
def reset_pellets():
    pellets.clear()
    for r,row in enumerate(MAZE):
        for c,ch in enumerate(row):
            if ch in '.o': pellets[(c,r)]=ch
reset_pellets()

def is_wall(c,r,door=False):
    if r<0 or r>=31: return True
    if c<0 or c>=28: return r!=14
    ch=MAZE[r][c]
    return ch=='#' or (ch=='-' and not door)

def draw_pellets(s,t):
    for (c,r),ch in pellets.items():
        x,y=cellPos(c,r)
        if ch=='.': pg.draw.rect(s,'peachpuff',(x-winScale,y-winScale,2*winScale,2*winScale))
        else:
            if((t//10)%2): continue 
            pg.draw.circle(s,'peachpuff',(x,y),3*winScale)
