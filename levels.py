MAX=0.15625 #100% speed, tiles per frame

def pick(l,t): #t=[(first_level,value),...] ascending
    v=t[0][1]
    for f,x in t:
        if l>=f: v=x
    return v

S1=[(420,'scatter'),(1200,'chase'),(420,'scatter'),(1200,'chase'),(300,'scatter'),(1200,'chase'),(300,'scatter'),(None,'chase')]
S2=[(420,'scatter'),(1200,'chase'),(420,'scatter'),(1200,'chase'),(300,'scatter'),(61980,'chase'),(1,'scatter'),(None,'chase')]
S5=[(300,'scatter'),(1200,'chase'),(300,'scatter'),(1200,'chase'),(300,'scatter'),(62220,'chase'),(1,'scatter'),(None,'chase')]

FRIGHT=[6,5,4,3,2,5,2,2,1,5,2,1,1,3,1,1,0,1] #seconds, levels 1-18, then 0
FLASH =[5,5,5,5,5,5,5,5,3,5,5,3,3,5,3,3,0,3]
FRUIT=[(1,('cherry',100)),(2,('strawberry',300)),(3,('orange',500)),(5,('apple',700)),
       (7,('melon',1000)),(9,('galaxian',2000)),(11,('bell',3000)),(13,('key',5000))]

def params(l):
    return dict(
        pac=pick(l,[(1,.80),(2,.90),(5,1.0),(21,.90)]),
        pac_fr=pick(l,[(1,.90),(2,.95),(5,1.0)]),
        gh=pick(l,[(1,.75),(2,.85),(5,.95)]),
        gh_tun=pick(l,[(1,.40),(2,.45),(5,.50)]),
        gh_fr=pick(l,[(1,.50),(2,.55),(5,.60)]),
        elroy=pick(l,[(1,20),(2,30),(3,40),(6,50),(9,60),(12,80),(15,100),(19,120)]),  #dots left for elroy 1, half for 2
        el_sp=(pick(l,[(1,.80),(2,.90),(5,1.0)]),pick(l,[(1,.85),(2,.95),(5,1.05)])),
        fright=FRIGHT[l-1] if l<=18 else 0,
        flash=FLASH[l-1] if l<=18 else 0,
        sched=pick(l,[(1,S1),(2,S2),(5,S5)]),
        fruit=pick(l,FRUIT),
        inky=pick(l,[(1,30),(2,0)]),
        clyde=pick(l,[(1,60),(2,50),(3,0)]),
        idle=pick(l,[(1,240),(5,180)]))  #frames without a dot before forced release