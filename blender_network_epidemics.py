import sys
sys.path.append("c:\\users\\strongi\\appdata\\roaming\\python\\python310\\site-packages") #replace the path with your site-package path

import bpy
from math import acos, degrees, pi
import math
from mathutils import Vector
import json
from random import choice, randrange, sample, random
from numpy import array,linspace,arange,linalg
import os
from math import *
from mathutils import *
import igraph
from colormaps import*
from collections import Counter

def initialize():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        for name in 'island', 'node', 'edge','x','y','ar','dot','tr','1850','moving_camera':
            if name in obj.name:
                obj.select_set(True)
    bpy.ops.object.delete(use_global=False)
    for i in range(5):
        bpy.ops.outliner.orphans_purge()
        bpy.ops.outliner.orphans_purge()
        bpy.ops.outliner.orphans_purge()


initialize()


#fast glass
def make_fast_glass(name,col=(0,0,0,1)):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.node_tree.nodes.remove(mat.node_tree.nodes["Principled BSDF"])
    glass_shader = mat.node_tree.nodes.new(type="ShaderNodeBsdfGlass")
    mixer = mat.node_tree.nodes.new(type="ShaderNodeMixShader")
    transparent_node = mat.node_tree.nodes.new(type="ShaderNodeBsdfTransparent")
    glass_shader.inputs[2].default_value = 0
    out = material_output = mat.node_tree.nodes.get("Material Output")
    glass_shader.distribution = 'GGX'
    mat.node_tree.links.new(transparent_node.outputs[0], mixer.inputs[1])
    mat.node_tree.links.new(glass_shader.outputs[0], mixer.inputs[2])
    mat.node_tree.links.new(mixer.outputs[0], out.inputs[0])
    mat.blend_method = 'BLEND'
    mat.node_tree.nodes["Transparent BSDF"].inputs[0].default_value = col



###NOW ADD THE CAMERA MOVEMENT
###function to make relative movements (camera moves towards object in movement)
def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'X')
    # assume we're using euler rotation
    rot0 = obj_camera.rotation_euler[2]
    obj_camera.rotation_euler = rot_quat.to_euler()
    if obj_camera.location[0]<0:
        obj_camera.rotation_euler[2] = rot0
        obj_camera.rotation_euler[0] = -obj_camera.rotation_euler[0]



def look_at_(obj, target, roll=0):
    """
    Rotate obj to look at target
    :arg obj: the object to be rotated. Usually the camera
    :arg target: the location (3-tuple or Vector) to be looked at
    :arg roll: The angle of rotation about the axis from obj to target in radians.
    """
    if not isinstance(target, Vector):
        target = Vector(target)
    loc = obj.location
    # direction points from the object to the target
    direction = target - loc
    tracker, rotator = (('-Z', 'Y'), 'Z') if obj.type == 'CAMERA' else (
    ('X', 'Z'), 'Y')  # because new cameras points down(-Z), usually meshes point (-Y)
    quat = direction.to_track_quat(*tracker)
    quat = quat.to_matrix().to_4x4()
    rollMatrix = Matrix.Rotation(roll, 4, rotator)
    loc = loc.to_tuple()
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc

def camera_chase(start, target, interp_steps,step):
    x0,y0,z0 = start
    xf,yf,zf = target
    xi,yi,zi = linspace(x0,xf,interp_steps),linspace(y0,yf,interp_steps),linspace(z0,zf,interp_steps)
    return (xi[step],yi[step],zi[step])

def animate_camera(node_seq,net,cam = 'Camera',tracker_vis = 'no',tot_steps=None,distance=1,start_step=1,def_interm_steps=50,move_away = 'yes',max_dist = 10,start_zoom_out_step=100):
    obj_camera = bpy.data.objects[cam]
    if tracker_vis=='yes':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=[0, 0, 0])
        bpy.context.active_object.name = 'tracker'
        tracker = bpy.data.objects['tracker']
        tracker.active_material = bpy.data.materials["red"]
    if not tot_steps:
        tot_steps = (len(node_seq)-1)*def_interm_steps
    seq_node_count = 0
    interm_steps = list(map(int,linspace(1,tot_steps,len(node_seq))))
    for step in range(1,int(tot_steps)+1):
        try:
            if step in interm_steps:
                node_start = net.vs['name'][node_seq[seq_node_count]]
                node_start_obj = bpy.data.objects[node_start]
                node_end = net.vs['name'][node_seq[seq_node_count+1]]
                node_end_obj = bpy.data.objects[node_end]
                seq_node_count+=1
            camera_target = Vector(camera_chase(node_start_obj.location,node_end_obj.location,interm_steps[seq_node_count]-interm_steps[seq_node_count-1],step-interm_steps[seq_node_count-1]))
            if tracker_vis=='yes':
                tracker.location = camera_target
            obj_camera.location = camera_target
            obj_camera.location[0] -= distance
            bpy.context.view_layer.update()
            look_at(obj_camera, camera_target)
            obj_camera.keyframe_insert(data_path="location", frame=start_step+step+1)
            obj_camera.keyframe_insert(data_path="rotation_euler", frame=start_step+step+1)
            obj_camera.keyframe_insert(data_path="rotation_axis_angle", frame=start_step+step+1)
            if tracker_vis == 'yes':
                tracker.keyframe_insert(data_path="location", frame=start_step+step+1)
        except:
            pass
        if move_away == 'yes' and step>=start_zoom_out_step:
            distance+=max_dist/(tot_steps-start_zoom_out_step)
    bpy.context.scene.frame_end = int(tot_steps)

#camera orbiting around target
#set your own target here
def camera_orbiting(target,r=1,rotations=1,step_start=0,num_step = 500,camera = 'Camera',x=0,y=1,rot=2):
    cam = bpy.data.objects['Camera']
    t_loc_x = target.location[x]
    t_loc_y = target.location[y]
    bpy.context.view_layer.update()
    look_at(cam, target.location)
    step = step_start
    cam.keyframe_insert(data_path="location", frame=step)
    cam.keyframe_insert(data_path="rotation_euler", frame=step)
    cam.keyframe_insert(data_path="rotation_axis_angle", frame=step)
    target_angle = rotations*(2*pi) # Go 90-8 deg more
    step+=1
    for s in range(num_step):
        alpha = (s)*target_angle/num_step
        cam.location[x] = t_loc_x+cos(alpha)*r
        cam.location[y] = t_loc_y+sin(alpha)*r
        look_at(cam, target.location)
        bpy.context.view_layer.update()
        cam.keyframe_insert(data_path="location", frame=step)
        cam.keyframe_insert(data_path="rotation_euler", frame=step)
        cam.keyframe_insert(data_path="rotation_axis_angle", frame=step)
        step+=1

#network functions

def make_sphere(name,node_size = 0.1,col=(0,0,0,1),glass='yes',add_sil='yes'):
    bpy.ops.mesh.primitive_uv_sphere_add(radius = 1,location = [0, 0, 0])
    bpy.context.active_object.name = name
    ob = bpy.data.objects[name]
    if glass=='yes':
        make_fast_glass(name+'_mat')
    else:
        bpy.data.materials.new(name+'_mat')
        bpy.data.materials[name+'_mat'].diffuse_color = col
    ob.active_material = bpy.data.materials[name+'_mat']
    mesh = ob.data
    for f in mesh.polygons:
         f.use_smooth = True
    if add_sil=='yes':
        bpy.ops.import_image.to_plane(files=[{'name': "human.png'}])
        sil = bpy.data.objects[str(sil_n)]
        sil.dimensions = (1.7,1.7,1.7)
        ctx = bpy.context.copy()
        ctx['active_object'] = ob
        ctx['selected_editable_objects'] = [ob,sil]
        bpy.ops.object.join(ctx)
    x,y,z = 10*random()*sample([-1,1],1)[0],10*random()*sample([-1,1],1)[0],10*random()*sample([-1,1],1)[0]
    ob.location = (x,y,z)
    ob.rotation_euler = (1-random()/10,1-random()/10,1-random()/10)
    ob.dimensions = [node_size] * 3
    return ob


def connect_nodes(edge_list, layout, edge_color="edge_mat", edge_thickness = 0.01, node_size=0.1,net_pre ='',animate='yes',
                  frame_start=1,duration=100):
    sc = 0
    edge_names = []
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
        edge_cylinder = bpy.data.objects.get(net_pre+'edge_'+str(sc))
        if edge_cylinder==None:
            bpy.ops.mesh.primitive_cylinder_add()
            edge_cylinder = bpy.context.object
            edge_cylinder.active_material = bpy.data.materials[edge_color]
            edge_cylinder.name = net_pre+'edge_'+str(sc)
            mesh = edge_cylinder.data
            for f in mesh.polygons:
                f.use_smooth = True
        edge_names.append(edge_cylinder.name)
        edge_cylinder.location = cent
        edge_cylinder.rotation_mode = "AXIS_ANGLE"
        edge_cylinder.rotation_axis_angle = [angle] + list(v_rot)
        edge_cylinder.keyframe_insert(data_path="location", frame=frame_start)
        if animate == 'yes':
            edge_cylinder.dimensions = (0,0,0)
            edge_cylinder.keyframe_insert(data_path="scale", frame=frame_start)
            edge_cylinder.dimensions = [edge_thickness] * 2 + [mag - node_size]
            edge_cylinder.keyframe_insert(data_path="scale", frame=frame_start+duration)
        else:
            edge_cylinder.dimensions = [edge_thickness] * 2 + [mag - node_size]
        sc += 1
    return edge_names


def move_nodes(layout,node_names='none',random_rot='no'):  ###assumes that nodes are numbered from 0 to n, with n = len(layout)
    if node_names=='none':
        node_names = ['node_'+str(i) for i in range(len(layout))]
    for i in range(len(layout)):
        node_i = bpy.data.objects.get(node_names[i])
        if node_i==None:
            node_i = make_sphere("node_"+str(i), add_sil='yes', node_size=0.2, col='glass')
        x,y,z = layout[i]
        node_i.location = (x,y,z)
        if random_rot!='no':
            node_i.rotation_euler[0] += random()
            node_i.rotation_euler[1] += random()
            node_i.rotation_euler[2] += random()


def layout_transition(edgelist,layout,new_layout,node_names,edge_names,start_step=1,interp_steps=250,interp_res=25,random_rot='no'):
    ###interpolate between initial and final layout, with loop
    interp_3d = [[linspace(layout[i][j],new_layout[i][j],interp_steps) for j in range(3)] for i in range(len(layout))]
    frame_n = start_step
    make_keyframes(frame_n, node_names+edge_names)
    for step in arange(0,interp_steps+interp_res,interp_res):
        if step>=len(interp_3d[0][0]):
            step = len(interp_3d[0][0])-1
        interp_layout = [[interp_3d[i][j][step] for j in range(3)] for i in range(len(layout))]
        move_nodes(interp_layout,random_rot = random_rot)
        edge_names = connect_nodes(edgelist, interp_layout,animate='no')
        make_keyframes(frame_n+1,node_names+edge_names)
        frame_n+=interp_res
    return frame_n


def move_edges(edge_list, layout,node_size=0.1,net_pre=''):
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
            edge_cylinder = bpy.data.objects[net_pre+'edge_'+str(edge_count)]
            edge_cylinder.dimensions[2] = mag - node_size
            edge_cylinder.location = cent
            edge_cylinder.rotation_mode = "AXIS_ANGLE"
            edge_cylinder.rotation_axis_angle = [angle] + list(v_rot)
        edge_count+=1


def make_keyframes(step,obj_names):
    for i in obj_names:
        obj = bpy.data.objects[i]
        obj.keyframe_insert(data_path="location", frame=step)
        obj.keyframe_insert(data_path="rotation_euler", frame=step)
        obj.keyframe_insert(data_path="rotation_axis_angle", frame=step)
        obj.keyframe_insert(data_path="scale", frame=step)


def make_all_keyframes(step,net_pre=''):
    ###nodes
    sc_node = 0
    obj = bpy.data.objects.get(net_pre+'node_'+str(sc_node))
    while obj!=None:
        obj.keyframe_insert(data_path="location", frame=step)
        obj.keyframe_insert(data_path="rotation_euler", frame=step)
        obj.keyframe_insert(data_path="rotation_axis_angle", frame=step)
        obj.keyframe_insert(data_path="scale", frame=step)
        sc_node+=1
        obj = bpy.data.objects.get(net_pre+'node_'+str(sc_node))
    #linls
    sc_edge = 0
    obj = bpy.data.objects.get(net_pre+'edge_'+str(sc_edge))
    while obj!=None:
        obj.keyframe_insert(data_path="location", frame=step)
        obj.keyframe_insert(data_path="rotation_euler", frame=step)
        obj.keyframe_insert(data_path="rotation_axis_angle", frame=step)
        obj.keyframe_insert(data_path="scale", frame=step)
        sc_edge+=1
        obj = bpy.data.objects.get(net_pre+'edge_'+str(sc_edge))




###identify path in network
def get_random_path(net):
    ##make arbitrary path
    n0 = randrange(len(net.vs))
    diam = [n0]
    n1 = 'no'
    while len(diam)<10 or n1!=diam[0]:
        n1 = sample(net.vs[n0].neighbors(),1)[0].index
        if len(diam)>1:
            while(n1==diam[-2]):
                n1 = sample(net.vs[n0].neighbors(), 1)[0].index
        diam.append(n1)
        n0 = n1



def norm_lay(layout,x0=0,y0=0,z0=0):
    dim = len(layout[0])
    if dim == 3:
        xs,ys,zs = [[(max(l)-i)/(max(l)-min(l))-0.5 for i in l] for l in zip(*layout)]
        xs = [3*i+x0 for i in xs]
        ys =[3*i+y0 for i in ys]
        zs =[3*i+z0 for i in zs]
        return (list(zip(xs,ys,zs)))
    else:
        ys, zs = [[(max(l) - i) / (max(l) - min(l)) - 0.5 for i in l] for l in zip(*layout)]
        ys = [3 * i + x0 for i in ys]
        zs = [3 * i + y0 for i in zs]
        xs = [0 for i in ys]
        return (list(zip(xs, ys,zs)))


def change_mat(name,frame,new_col=(1,0,0,1),duration=1,speed=1,highlight=False):
    tot_frames = duration*bpy.context.scene.render.fps
    if tot_frames<1:
        tot_frames=1
    if bpy.data.materials[name + '_mat'].node_tree:
        bpy.data.materials[name + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].keyframe_insert('default_value', frame = frame-1)
        orig_col = bpy.data.materials[name + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].default_value[:]
        cur_col = orig_col
        for fr_n in arange(frame,frame+tot_frames,int(round(10/speed))):
            cur_col = new_col if cur_col!=new_col else orig_col
            bpy.data.materials[name + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].default_value = cur_col
            bpy.data.materials[name + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].keyframe_insert('default_value',frame=fr_n)
        if highlight==False:
            bpy.data.materials[name + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].default_value = new_col
        else:
            bpy.data.materials[name + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].default_value = orig_col
        bpy.data.materials[name + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].keyframe_insert('default_value',frame=frame+tot_frames)
    else:
        bpy.data.materials[name + '_mat'].keyframe_insert(data_path='diffuse_color', frame = frame-1)
        orig_col = bpy.data.materials[name + '_mat'].diffuse_color
        cur_col = orig_col
        for fr_n in arange(frame,frame+tot_frames,int(round(10/speed))):
            cur_col = new_col if cur_col!=new_col else orig_col
            bpy.data.materials[name + '_mat'].diffuse_color = cur_col
            bpy.data.materials[name + '_mat'].keyframe_insert(data_path='diffuse_color', frame=fr_n)
        if highlight==False:
            bpy.data.materials[name + '_mat'].diffuse_color = new_col
        else:
            bpy.data.materials[name + '_mat'].diffuse_color = orig_col
        bpy.data.materials[name + '_mat'].keyframe_insert(data_path='diffuse_color', frame=frame+tot_frames)



def epidemic(g,p_inf=0.1,max_count=10000,generate_tracker='yes',start_frame=1,duration=5):
    if generate_tracker == 'yes':
        for tr_n in range(len(g.vs)):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(1000,1000,1000))
            bpy.context.active_object.name = 'tr_' + str(tr_n)
            tracker = bpy.data.objects['tr_' + str(tr_n)]
            tracker.active_material = bpy.data.materials["tracker_mat"]
            mesh = tracker.data
            for f in mesh.polygons:
                f.use_smooth = True
            bpy.ops.import_image.to_plane(files=[{'name': "virus.png"}])
            sil = bpy.data.objects["1850"]
            sil.dimensions = (0.05, 0.05,0.05)
            sil.location = (1000,1000,1000)
            sil.rotation_euler = [math.radians(90),math.radians(360),math.radians(90)]
            ctx = bpy.context.copy()
            ctx['active_object'] = tracker
            ctx['selected_editable_objects'] = [tracker, sil]
            bpy.ops.object.join(ctx)
    g.vs['s'] = 'S'
    i0 = randrange(len(g.vs))
    seq_dict,seq_list= dict([]),[[[i0,i0]]]
    g.vs[i0]['s'] = 'I'
    count = 0
    while 'S' in g.vs['s'] and count<max_count:
        inf_step = []
        for s_node in [k for k in g.vs if k['s'] == 'I']:
            for t_node in s_node.neighbors():
                if t_node['s']!='I' and random()<p_inf:
                    seq_dict[t_node.index] = s_node.index
                    inf_step.append([s_node.index,t_node.index])
                    t_node['s'] = 'I'
        seq_list.append(inf_step)
        count+=1
    i_last = list(seq_dict.keys())[-1]
    i_seq = [i_last]
    while (seq_dict.get(i_seq[-1], None) != None):
        i_seq.append(seq_dict[i_seq[-1]])
    i_seq = i_seq[::-1]
    tot_steps = len(seq_list) * 25
    tr_n = 0
    step = 0
    for i in seq_list:
        for j in i:
            node_start_loc = bpy.data.objects[g.vs['name'][j[0]]].location
            tracker = bpy.data.objects['tr_' + str(tr_n)]
            tracker.keyframe_insert(data_path="location", frame=start_frame-1)
            tracker.location = node_start_loc
            tracker.keyframe_insert(data_path="location", frame=start_frame)
            change_mat(g.vs[j[1]]['name'], start_frame + tot_steps / len(seq_list), (1, 0, 0, 1), duration=duration,
                       speed=10)
            node_end_loc = bpy.data.objects[g.vs['name'][j[1]]].location
            tracker.location = node_end_loc
            tracker.keyframe_insert(data_path="location", frame=start_frame + tot_steps / len(seq_list))
            tracker.location = (1000,1000,1000)
            tracker.keyframe_insert(data_path="location", frame=1+start_frame + tot_steps / len(seq_list))
            tr_n += 1
        start_frame += tot_steps / len(seq_list)
    return(seq_list,seq_dict,start_frame,i_seq)

###plot function

def rescale_01(x):
    min_x,max_x = min(x),max(x)
    d = max_x-min_x
    return ([(i-min_x)/d for i in x])


def make_plot(data_x,data_y,x_min=0,y_min=0,x_max=1,y_max=1,or_x = 0,or_y = 0, or_z = 0,n_ticks = 5,
    nround =1,make_axes='yes',plot_id='',dot_col = 'red',axes_col = 'black',animate='yes',frame_start=0,duration=100,make_camera='no',reset='yes',xlabel='xlabel',ylabel='ylabel',used_dots=0):
    if plot_id == '':
        plot_id = 0
        while bpy.data.objects.get('x_'+str(plot_id)) != None:
            plot_id+=1
        plot_id = str(plot_id)+'_'
    if reset=='yes':
        for obj in bpy.context.scene.objects:
            if 'dot' in obj.name:
                obj.keyframe_insert(data_path="location", frame=frame_start)
                obj.location=(1000, 1000, 1000)
                obj.keyframe_insert(data_path="location", frame=frame_start+1)
    if make_axes=='yes':
        bpy.ops.mesh.primitive_cube_add()
        bpy.context.active_object.name = plot_id+'x'
        ob = bpy.data.objects[plot_id+'x']
        ob.active_material = bpy.data.materials[axes_col]
        axis_length = 1.1
        ob.dimensions = [0.02,axis_length,0.02]
        ob.location = [or_x,or_y-axis_length/2,or_z]
        bpy.ops.mesh.primitive_cylinder_add(vertices = 3, radius = 0.3, depth = 0.1)
        bpy.context.active_object.name = plot_id+'ar_x'
        cyl1 = bpy.data.objects[plot_id+'ar_x']
        cyl1.location = [or_x, or_y-axis_length, or_z]
        cyl1.scale = [0.1,0.17,0.1]
        cyl1.rotation_euler = [0,math.radians(90),math.radians(180)]
        cyl1.active_material = bpy.data.materials[axes_col]
        # Add z-axis and set its dimensions
        bpy.ops.mesh.primitive_cube_add()
        bpy.context.active_object.name = plot_id+'y'
        ob = bpy.data.objects[plot_id+'y']
        ob.active_material = bpy.data.materials[axes_col]
        axis_height = 1.1
        ob.dimensions = [0.02,0.02,axis_height]
        ob.location = [or_x,or_y,or_z+axis_height/2]
        bpy.ops.mesh.primitive_cylinder_add(vertices = 3, radius = 0.3, depth = 0.1)
        bpy.context.active_object.name = plot_id+'ar_y'
        cyl2 = bpy.data.objects[plot_id+'ar_y']
        cyl2.location = [or_x, or_y, or_z+axis_height]
        cyl2.scale = [0.1,0.17,0.1]
        cyl2.rotation_euler = [math.radians(90),0,math.radians(90)]
        cyl2.active_material = bpy.data.materials[axes_col]
        ###make labels
        pos_labs = linspace(0,1,n_ticks+1)
        pos_x = 0
        for i in linspace(x_min,x_max,n_ticks+1):
            bpy.ops.object.text_add()
            bpy.context.active_object.name = plot_id+'x_lab_'+str(pos_x)
            text = bpy.context.active_object
            text.data.body = str(round(i,nround))
            bpy.ops.object.convert(target="MESH")
            text.scale = (0.12,0.12,0.12)
            text_width,text_height,text_tick = 0.12*(text.dimensions)
            text.location = (or_x,or_y-(pos_labs[pos_x]-text_width/2),or_z-(text_height+0.05))
            text.active_material = bpy.data.materials[axes_col]
            text.rotation_euler = [math.radians(90),0,math.radians(-90)]
            pos_x+=1
        pos_y = 0
        for i in linspace(y_min,y_max,n_ticks+1):
            bpy.ops.object.text_add()
            bpy.context.active_object.name = plot_id+'y_lab_'+str(pos_y)
            text = bpy.context.active_object
            text.data.body = str(round(i,nround))
            bpy.ops.object.convert(target="MESH")
            text.active_material = bpy.data.materials[axes_col]
            text.rotation_euler = [math.radians(90),0,math.radians(-90)]
            text.scale = (0.12,0.12,0.12)
            text_width,text_height,text_tick = 0.12*text.dimensions#
            text.location = (or_x,or_y+(text_width+0.05),or_z+pos_labs[pos_y]-text_height/2)
            pos_y+=1
        ###x label
        bpy.ops.object.text_add()
        bpy.context.active_object.name = plot_id + 'x_label'
        text = bpy.context.active_object
        text.data.body = xlabel
        bpy.ops.object.convert(target="MESH")
        text.scale = (0.25, 0.25, 0.25)
        text_width, text_height, text_tick = 0.25 * (text.dimensions)
        text.location = (or_x, or_y - (axis_length/2 - text_width/2), or_z - (text_height + 0.15))
        text.active_material = bpy.data.materials[axes_col]
        text.rotation_euler = [math.radians(90), 0, math.radians(-90)]
        ###y label
        bpy.ops.object.text_add()
        bpy.context.active_object.name = plot_id + 'y_label'
        text = bpy.context.active_object
        text.data.body = ylabel
        bpy.ops.object.convert(target="MESH")
        text.scale = (0.25, 0.25, 0.25)
        text_width, text_height, text_tick = 0.25 * (text.dimensions)
        text.location = (or_x, or_y + (text_height + 0.15), or_z + axis_height/2 - text_width/2)
        text.active_material = bpy.data.materials[axes_col]
        text.rotation_euler = [math.radians(90), math.radians(-90), math.radians(-90)]
    ###plot the points
    used_dots = used_dots
    for i in range(len(data_x)):
        name = plot_id+'dot_'+str(i+used_dots)
        ob = bpy.data.objects.get(name,None)
        if ob==None:
            bpy.ops.mesh.primitive_uv_sphere_add(radius = 0.025,location = (1000, 1000, 1000))
            bpy.context.active_object.name = name
            ob = bpy.data.objects.get(name,None)
            ob.keyframe_insert(data_path="location", frame=0)
        ob.active_material = bpy.data.materials[dot_col]
        ob.location = (or_x, or_y-data_x[i], or_z + data_y[i])
        ob.keyframe_insert(data_path="location", frame=frame_start + 2+int(round(i * duration / len(data_x))))
    ###point the camera
    scn = bpy.context.scene
    # create camera
    if make_camera == 'yes':
        cam1 = bpy.data.cameras.new(plot_id+'_camera_plot')
        cam1.lens = 18
        # create the first camera object
        cam_obj1 = bpy.data.objects.new(plot_id+'_camera_plot', cam1)
        cam_obj1.location=(or_x+0.5, -2, or_z+0.5)
        cam_obj1.rotation_euler = (math.radians(90),0,0)
        scn.collection.objects.link(cam_obj1)




def add_text(text_content,text_id,frame_start,frame_end,position,size = 0.5,text_col='black'):
    x,y,z = position
    bpy.ops.object.text_add()
    bpy.context.active_object.name = text_id
    text = bpy.context.active_object
    text.data.body = text_content
    bpy.ops.object.convert(target="MESH")
    text.scale = (size, size, size)
    text_width, text_height, text_tick = size * (text.dimensions)
    text.active_material = bpy.data.materials[text_col]
    text.rotation_euler = [math.radians(90), 0, math.radians(-90)]
    text.location = (1000,1000,1000)
    text.keyframe_insert(data_path="location", frame=1)
    text.keyframe_insert(data_path="location", frame=frame_start-1)
    text.keyframe_insert(data_path="location", frame=frame_end+1)
    text.location = (x, y + text_width/2, z)
    text.keyframe_insert(data_path="location", frame=frame_start)
    text.keyframe_insert(data_path="location", frame=frame_end)


def create_network(network,layout,new_network='',new_layout='',position=(0,0,0),net_pre='',frame_start=1,duration=0,reset_duration=1,node_cols=[],glass='yes',col=(0,0,0,1),animate_edges='yes'):
    x0,y0,z0 = position
    node_n = len(network.vs)
    e_n = len(network.es)
    node_names = [net_pre + 'node_' + str(i) for i in range(node_n)]
    edge_names = [net_pre + 'edge_' + str(i) for i in range(e_n)]
    if new_network=='':
        edge_list = [i.tuple for i in network.es]
        col_n = 0
        for node_id in node_names:
            if node_cols!=[]:
                make_sphere(node_id,add_sil='yes',node_size = 0.2,col=node_cols[col_n],glass=glass)
                col_n+=1
            else:
                make_sphere(node_id,add_sil='yes',node_size = 0.2,col=col,glass=glass)
            if glass == 'yes':
                bpy.data.materials[node_id + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].keyframe_insert(
                    'default_value', frame=frame_start)
            else:
                bpy.data.materials[node_id + '_mat'].keyframe_insert(data_path='diffuse_color', frame=frame_start)
        make_keyframes(frame_start,node_names)
        move_nodes(layout, node_names,random_rot='yes')
        edge_names = connect_nodes(edge_list,layout,edge_thickness=0.01,node_size = 0.2,net_pre = '',frame_start=frame_start+int(duration/2),duration=int(duration/2),animate=animate_edges)
        make_keyframes(frame_start+duration,node_names)
        return (frame_start + duration, network, layout)
    else:
        new_edgelist = [i.tuple for i in new_network.es]
        # reset the node color
        col_n = 0
        for i in [obj.name for obj in bpy.context.scene.objects if 'node' in obj.name]:
            if node_cols!=[]:
                col = node_cols[col_n]
                col_n+=1
                if col_n==len(node_cols):
                    col_n = len(node_cols)-1
            if bpy.data.materials[i + '_mat'].node_tree:
                bpy.data.materials[i + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].keyframe_insert('default_value',frame=frame_start-1)
                bpy.data.materials[i + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].default_value = col
                bpy.data.materials[i + '_mat'].node_tree.nodes["Transparent BSDF"].inputs[0].keyframe_insert('default_value',frame=frame_start+reset_duration)
            else:
                bpy.data.materials[i + '_mat'].keyframe_insert(data_path='diffuse_color', frame=frame_start-1)
                bpy.data.materials[i + '_mat'].diffuse_color = col
                bpy.data.materials[i + '_mat'].keyframe_insert(data_path='diffuse_color', frame=frame_start+reset_duration)
        frame_start+=reset_duration
        tot_steps = layout_transition(new_edgelist, layout, new_layout, node_names, edge_names, frame_start,interp_steps=duration+50,random_rot='yes')
        tot_steps += 50
        return (tot_steps,new_network,new_layout)


def match(network,new_network):
    node_n = len(network.vs)
    e_n = len(network.es)
    while len(new_network.es)>e_n:
        r_e = sample(range(len(new_network.es)),1)[0]
        v1,v2 = new_network.es[r_e].tuple
        if new_network.vs[v1].degree()>1 and new_network.vs[v1].degree()>1:
            new_network.delete_edges(r_e)
    return (new_network)


def get_dd(g):
    deg = g.degree()
    dd = Counter(deg)
    x = list(range(max(deg)))
    y = [dd[i] for i in x]
    return (x,y)






initialize()

make_fast_glass('edge_mat',col=(0.5,0.5,0.5,1))

make_fast_glass('tracker_mat',col=(0.5, 0, 0.3, 1))


###make some materials
colors = {"purple": (178, 132, 234), "gray": (11, 11, 11),
          "green": (114, 195, 0), "red": (255, 0, 75),
          "blue": (0, 131, 255), "clear": (0, 131, 255),
          "yellow": (255, 187, 0), "light_gray": (118, 118, 118),
          "white": (255, 255, 255), "black":(0,0,0)}

# Normalize to [0,1] and make blender materials
for key, value in colors.items():
    value = [x / 255.0 for x in value] + [1]
    bpy.data.materials.new(name=key)
    bpy.data.materials[key].diffuse_color = value





delay = 100
duration = 300
tot_steps = 1
###make the initial network
network = igraph.Graph.Lattice([10,10], nei=1, directed=False, mutual=True, circular=False)
layout = norm_lay(list(network.layout(layout='grid',dim=2)))
node_n = len(network.vs)
e_n = len(network.es)
data_x,data_y = get_dd(network)
data_x = rescale_01(data_x)
data_y = rescale_01(data_y)

node_cols = [viridis(i/max(network.degree())) for i in network.degree()]
tot_steps,network,layout = create_network(network,layout,new_network='',new_layout='',position=(0,0,0),net_pre='',frame_start=tot_steps+delay,duration=100,reset_duration=1,glass='yes',node_cols=[],col=(0,1,0,1))
add_text("regular lattice","text1",delay,tot_steps+delay,position=(0,0,1.72),size = 0.4,text_col='black')

for obj in bpy.context.scene.objects:
    if 'node' in obj.name:
        obj.rotation_euler = (0, 0, 0)
        obj.keyframe_insert(data_path="rotation_euler", frame=tot_steps)

node_names = ['node_'+str(i) for i in range(node_n)]
network.vs()['name'] = node_names
t0 = tot_steps
seq_list,seq_dict,tot_steps,i_seq = epidemic(network,start_frame=tot_steps)
data_x = list(range(len(seq_list)))
data_y = [sum([len(i) for i in seq_list[:j]]) for j in range(len(seq_list))]
data_x = rescale_01(data_x)
data_y = rescale_01(data_y)
duration_ep_lattice = int(tot_steps-t0)

make_plot(data_x,data_y,or_x = 0, or_y = 3.2, or_z = 0.8,plot_id='1_',frame_start=tot_steps-duration_ep_lattice,duration=duration_ep_lattice,make_axes='yes',xlabel='time',ylabel='infections')

# create the 2nd Camera
cam = bpy.data.cameras.new("moving_camera")
cam_obj = bpy.data.objects.new("moving_camera", cam)
cam_obj.location = (-50, 0, 0)
scn = bpy.context.scene
scn.collection.objects.link(cam_obj)
look_at(cam_obj, Vector([0,0,0]))
animate_camera(i_seq,network,cam="moving_camera",start_step=t0,tot_steps=duration_ep_lattice,move_away='no',distance=3,max_dist=30)



new_network = igraph.Graph.Barabasi(node_n,int(round(e_n/node_n)))
new_network = match(network,new_network)
while len(network.components())>1:
    new_network = igraph.Graph.Barabasi(node_n, int(round(e_n / node_n)))
    new_network = match(network, new_network)

tot_steps+=50


node_names = ['node_'+str(i) for i in range(len(new_network.vs()))]
new_layout = norm_lay(list(new_network.layout(layout='fr3d')))
tot_steps,network,layout = create_network(network,layout,new_network=new_network,new_layout=new_layout,position=(0,0,0),net_pre='',frame_start=tot_steps,duration=200,reset_duration=100,glass='yes',node_cols=[],col=(0,0,0,1))
network.vs()['name'] = node_names
tot_steps+=10
t0 = tot_steps
for obj in bpy.context.scene.objects:
    if 'node' in obj.name:
        obj.rotation_euler = (0, 0, 0)
        obj.keyframe_insert(data_path="rotation_euler", frame=tot_steps)

seq_list,seq_dict,tot_steps,i_seq = epidemic(network,generate_tracker='no',start_frame=tot_steps)
duration_ep_lattice = int(tot_steps-t0)

data_y = [sum([len(i) for i in seq_list[:j]]) for j in range(len(seq_list))]
used_dots =len(data_x)
data_x = data_x[:len(seq_list)]
data_y = rescale_01(data_y)

make_plot(data_x,data_y,or_x = 0, or_y = 3.2, or_z = 0.8,plot_id='2_',frame_start=t0,duration=duration_ep_lattice,make_axes = 'no',reset='no',used_dots=used_dots,dot_col='blue',make_camera='no')

animate_camera(i_seq,network,cam="moving_camera",start_step=t0,tot_steps=duration_ep_lattice,move_away='no',distance=3,max_dist=30)

tot_steps+=200

