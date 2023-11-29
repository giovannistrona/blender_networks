import sys
sys.path.append("c:\\users\\strongi\\appdata\\roaming\\python\\python310\\site-packages")

import bpy
from math import acos, degrees, pi
from mathutils import Vector
import json
from random import choice, randrange, sample, random
from numpy import array,linspace,arange
import os
from math import *
from mathutils import *
import igraph

def initialize():
    for i in range(5):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.outliner.orphans_purge()
        bpy.ops.outliner.orphans_purge()
        bpy.ops.outliner.orphans_purge()


initialize()

bpy.context.scene.frame_end = 500

bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 200),
                          rotation=(1.5728, 1.09178e-09, 0.0689896), scale=(1, 1, 1))


def make_nodes(n,node_color="blue",node_size = 0.1):
    colors = {"purple": (178, 132, 234), "gray": (11, 11, 11),
              "green": (114, 195, 0), "red": (255, 0, 75),
              "blue": (0, 131, 255), "clear": (0, 131, 255),
              "yellow": (255, 187, 0), "light_gray": (118, 118, 118)}
    # Normalize to [0,1] and make blender materials
    for key, value in colors.items():
        value = [x / 255.0 for x in value] + [1]
        bpy.data.materials.new(name=key)
        bpy.data.materials[key].diffuse_color = value
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_uv_sphere_add()
    sphere = bpy.context.object
    sphere.dimensions = [0.1] * 3
    for i in range(n):  # key, node in network["nodes"].items():
        node_sphere = sphere.copy()
        node_sphere.data = sphere.data.copy()
        node_sphere.name = 'node_'+str(i)
        loc = tuple([randrange(-50, 50), randrange(-50, 50), randrange(-50, 50)])
        node_sphere.location = loc
        node_sphere.dimensions = [node_size] * 3
        node_sphere.active_material = bpy.data.materials[node_color]
        bpy.context.collection.objects.link(node_sphere)
    bpy.ops.object.select_all(action='DESELECT')
    sphere.select_set(True)
    bpy.ops.object.delete()



def connect_nodes(edge_list, layout, edge_color = "light_gray", edge_thickness = 0.025, node_size=0.1):
    bpy.ops.mesh.primitive_cylinder_add()
    cylinder = bpy.context.object
    cylinder.active_material = bpy.data.materials[edge_color]
    sc = 0
    for source_node,target_node in edge_list:
        source_loc = array(layout[source_node])
        target_loc = array(layout[target_node])
        diff = [c2 - c1 for c2, c1 in zip(source_loc, target_loc)]
        cent = [(c2 + c1) / 2 for c2, c1 in zip(source_loc, target_loc)]
        mag = sum([(c2 - c1) ** 2 for c1, c2 in zip(source_loc, target_loc)]) ** 0.5
        # Euler rotation calculation
        v_axis = Vector(diff).normalized()
        v_obj = Vector((0, 0, 1))
        v_rot = v_obj.cross(v_axis)
        angle = acos(v_obj.dot(v_axis))
        # Copy mesh primitive to create edge
        edge_cylinder = cylinder.copy()
        edge_cylinder.data = cylinder.data.copy()
        edge_cylinder.name = 'edge_'+str(sc)
        sc+=1
        edge_cylinder.dimensions = [edge_thickness] * 2 + [mag - node_size]
        edge_cylinder.location = cent
        edge_cylinder.rotation_mode = "AXIS_ANGLE"
        edge_cylinder.rotation_axis_angle = [angle] + list(v_rot)
        bpy.context.collection.objects.link(edge_cylinder)
    bpy.ops.object.select_all(action='DESELECT')
    cylinder.select_set(True)
    bpy.ops.object.delete()


def move_nodes(layout):  ###assumes that nodes are numbered from 0 to n, with n = len(layout)
    for i in range(len(layout)):
        node_i = bpy.data.objects['node_'+str(i)]
        x,y,z = layout[i]
        node_i.location = (x,y,z)


def move_edges(edge_list, layout,node_size=0.1):
    edge_count = 0
    for source_node,target_node in edge_list:
        if "removed" not in [source_node,target_node]:
            source_loc = array(layout[source_node])
            target_loc = array(layout[target_node])
            diff = [c2 - c1 for c2, c1 in zip(source_loc, target_loc)]
            cent = [(c2 + c1) / 2 for c2, c1 in zip(source_loc, target_loc)]
            mag = sum([(c2 - c1) ** 2 for c1, c2 in zip(source_loc, target_loc)]) ** 0.5
            # Euler rotation calculation
            v_axis = Vector(diff).normalized()
            v_obj = Vector((0, 0, 1))
            v_rot = v_obj.cross(v_axis)
            angle = acos(v_obj.dot(v_axis))
            # Copy mesh primitive to create edge
            edge_cylinder = bpy.data.objects['edge_'+str(edge_count)]
            edge_cylinder.dimensions[2] = mag - node_size
            edge_cylinder.location = cent
            edge_cylinder.rotation_mode = "AXIS_ANGLE"
            edge_cylinder.rotation_axis_angle = [angle] + list(v_rot)
        edge_count+=1



def make_keyframes(step,node_names,edge_names):
    for i in node_names+edge_names:
        obj = bpy.data.objects[i]
        obj.keyframe_insert(data_path="location", frame=step)
        obj.keyframe_insert(data_path="rotation_euler", frame=step)
        obj.keyframe_insert(data_path="rotation_axis_angle", frame=step)
        obj.keyframe_insert(data_path="scale", frame=step)


network = igraph.Graph.Barabasi(100,4,directed=False)
node_names = ['node_'+str(i) for i in range(len(network.vs))]
edge_names = ['edge_'+str(i) for i in range(len(network.es))]
network.vs['name'] = node_names
network.es['name'] = edge_names
layout = list(network.layout(layout='fr3d'))
edge_list = [i.tuple for i in network.es]
make_nodes(len(network.vs),node_size=0.5)
move_nodes(layout)
connect_nodes(edge_list,layout,edge_thickness=0.01)
make_keyframes(1,node_names,edge_names)


###interpolate between initial and final layout, with loop
layout_final = list(network.layout(layout='fr3d'))
interp_steps = 500
interp_3d = [[linspace(layout[i][j],layout_final[i][j],interp_steps) for j in range(3)] for i in range(len(layout))]
for step in arange(1,interp_steps/2,25):
    interp_layout = [[interp_3d[i][j][int(step)-1] for j in range(3)] for i in range(len(node_names))]
    move_nodes(interp_layout)
    move_edges(edge_list, interp_layout,node_size=0.5)
    make_keyframes(step,node_names,edge_names)

from numpy import arange
interp_3d = [[linspace(layout_final[i][j],layout[i][j],interp_steps) for j in range(3)] for i in range(len(layout))]
for step in arange(interp_steps/2,interp_steps+25,25):
    interp_layout = [[interp_3d[i][j][int(step)-1] for j in range(3)] for i in range(len(node_names))]
    move_nodes(interp_layout)
    move_edges(edge_list, interp_layout,node_size=0.5)
    make_keyframes(step,node_names,edge_names)






