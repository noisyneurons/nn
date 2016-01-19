#temp
from __future__ import division
import math

assert len([2,4]) == 3

def rotatePolygon(polygon,theta):
    """Rotates the given polygon which consists of corners represented as (x,y),
    around the ORIGIN, clock-wise, theta degrees"""
    theta = math.radians(theta)
    rotatedPolygon = []
    for corner in polygon :
        rotatedPolygon.append(( corner[0]*math.cos(theta)-corner[1]*math.sin(theta) , corner[0]*math.sin(theta)+corner[1]*math.cos(theta)) )
    return rotatedPolygon


#my_polygon = [(0,0),(1,0),(0,1)]
my_polygon = [[0,0],[1,0],[0,1]]

print rotatePolygon(my_polygon,45)

