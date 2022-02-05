from shapely.geometry import LineString, Point
import math

#####################################
## Classification dataset MapInfo ###
#####################################
corner_offset = 0.1

SA1_cor_left = -2.21
SA1_cor_right = 2.21
SA1_front_left = 6.63
SA1_front_right = 6.63
SA1_wall = SA1_front_left+8.00
SA1_width = 12.5
SA1_map_info = {
    "width" : 2*SA1_width,
    "height" : 20,
    "walls" : [LineString([(SA1_cor_left,0),(SA1_cor_left,SA1_front_left)]), 
            LineString([(SA1_cor_right,0),(SA1_cor_right,SA1_front_right)]),
            LineString([(-SA1_width,SA1_wall),(SA1_width,SA1_wall)]),
            LineString([(SA1_cor_left,SA1_front_left),(-SA1_width,SA1_front_left)]),
            LineString([(SA1_cor_right,SA1_front_right),(SA1_width,SA1_front_right)])],
    "corners" : [Point(SA1_cor_left+corner_offset,SA1_front_left+corner_offset), Point(SA1_cor_right-corner_offset,SA1_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "front_range" : [SA1_cor_left, SA1_cor_right],
    "road_range" : [2*SA1_width, SA1_wall-SA1_front_left-0.2, SA1_front_left+0.1]
}

SA2_cor_left = -3.06
SA2_cor_right = 3.46
SA2_front_left = 8.21
SA2_front_right = 8.21
SA2_wall = SA2_front_left+8.24
SA2_width = 12.5
SA2_map_info = {
    "width" : 2*SA2_width,
    "height" : 20,
    "walls" : [LineString([(SA2_cor_left,0),(SA2_cor_left,SA2_front_left)]), 
            LineString([(SA2_cor_right,0),(SA2_cor_right,SA2_front_right)]),
            LineString([(-SA2_width,SA2_wall),(SA2_width,SA2_wall)]),
            LineString([(SA2_cor_left,SA2_front_left),(-SA2_width,SA2_front_left)]),
            LineString([(SA2_cor_right,SA2_front_right),(SA2_width,SA2_front_right)])],
    "corners" : [Point(SA2_cor_left+corner_offset,SA2_front_left+corner_offset), Point(SA2_cor_right-corner_offset,SA2_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "front_range" : [SA2_cor_left, SA2_cor_right],
    "road_range" : [2*SA2_width, SA2_wall-SA2_front_left-0.2, SA2_front_left+0.1]
}

SB1_cor_left = -2.06
SB1_cor_right = 3.80
SB1_front_left = 6.37
SB1_front_right = 6.37
SB1_width = 12.5
SB1_map_info = {
    "width" : 2*SB1_width,
    "height" : 20,
    "walls" : [LineString([(SB1_cor_left,0),(SB1_cor_left,SB1_front_left)]), 
            LineString([(SB1_cor_right,0),(SB1_cor_right,SB1_front_right)]),
            LineString([(SB1_cor_left,SB1_front_left),(-SB1_width,SB1_front_left)]),
            LineString([(SB1_cor_right,SB1_front_right),(SB1_width,SB1_front_right)])],
    "corners" : [Point(SB1_cor_left+corner_offset,SB1_front_left+corner_offset), Point(SB1_cor_right-corner_offset,SB1_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "front_range" : [SB1_cor_left, SB1_cor_right],
    "road_range" : [2*SB1_width, 20-SB1_front_left-0.2, SB1_front_left+0.1]
}

SB2_cor_left = -4.79
SB2_cor_right = 4.38
SB2_front_left = 8.18
SB2_front_right = 8.18
SB2_width = 12.5
SB2_map_info = {
    "width" : 2*SB2_width,
    "height" : 20,
    "walls" : [LineString([(SB2_cor_left,0),(SB2_cor_left,SB2_front_left)]), 
            LineString([(SB2_cor_right,0),(SB2_cor_right,SB2_front_right)]),
            LineString([(SB2_cor_left,SB2_front_left),(-SB2_width,SB2_front_left)]),
            LineString([(SB2_cor_right,SB2_front_right),(SB2_width,SB2_front_right)])],
    "corners" : [Point(SB2_cor_left+corner_offset,SB2_front_left+corner_offset), Point(SB2_cor_right-corner_offset,SB2_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "front_range" : [SB2_cor_left, SB2_cor_right],
    "road_range" : [2*SB2_width, 16.5-SB2_front_left-0.2, SB2_front_left+0.1]
}

SB3_cor_left = -4.04
SB3_cor_right = 3.11
SB3_front_left = 7.37
SB3_front_right = 7.37
SB3_width = 12.5
SB3_map_info = {
    "width" : 2*SB3_width,
    "height" : 20,
    "walls" : [LineString([(SB3_cor_left,0),(SB3_cor_left,SB3_front_left)]), 
            LineString([(SB3_cor_right,0),(SB3_cor_right,SB3_front_right)]),
            LineString([(SB3_cor_left,SB3_front_left),(-SB3_width,SB3_front_left)]),
            LineString([(SB3_cor_right,SB3_front_right),(SB3_width,SB3_front_right)])],
    "corners" : [Point(SB3_cor_left+corner_offset,SB3_front_left+corner_offset), Point(SB3_cor_right-corner_offset,SB3_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "front_range" : [SB3_cor_left, SB3_cor_right],
    "road_range" : [2*SB3_width, 13-SB3_front_left-0.2, SB3_front_left+0.1]
}