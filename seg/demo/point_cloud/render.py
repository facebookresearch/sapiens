import os
import re
import bpy
import sys
import argparse
import numpy as np
from math import radians
from mathutils import Matrix, Quaternion

def set_blender_gpu():
    print('--------------------------setting up gpu--------------------------')
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # Or 'OPENCL' or 'OPTIX'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.feature_set = 'SUPPORTED'

    for scene in bpy.data.scenes:
        print(scene.name)
        scene.cycles.device = 'GPU'

    # List and enable GPU devices
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    for device in cycles_prefs.devices:
        device.use = True
        print(f"Device: {device.name}, Type: {device.type}, Use: {device.use}")

    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d.use = True
        if d.type == 'CPU':
            d.use = False
        print("Device '{}' type {} : {}".format(d.name, d.type, d.use))
    print('-----------------------setting up gpu done-----------------------')
    return

class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())

## in rgb
def get_colors(c):
    colors = {
        'pink': np.array([197, 27, 125]),
        'light_pink': np.array([233, 163, 201]),
        'light_green': np.array([161, 215, 106]),
        'green': np.array([77, 146, 33]),
        'red': np.array([215, 48, 39]),
        'medium_red': np.array([237, 85, 59]),
        'light_red': np.array([252, 146, 114]),
        'light_orange': np.array([252, 141, 89]),
        'purple': np.array([118, 42, 131]),
        'medium_purple': np.array([142, 69, 173]),  
        'light_purple': np.array([175, 141, 195]),
        'light_blue': np.array([145, 191, 219]),
        'blue': np.array([69, 117, 180]),
        'gray': np.array([130, 130, 130]),
        'white': np.array([255, 255, 255]),
        'turkuaz': np.array([50, 134, 204]),
        'yellow': np.array([235, 219, 52]),
        # 'orange': np.array([242, 151, 43]),
        'orange': np.array([220, 88, 42]),

    }
    return colors[c]

##################################################
# Helper functions
##################################################

# Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
#   Source: smpl/plugins/blender/corrective_bpy_sh.py
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)

def np_array_from_image(img_name):
    img = bpy.data.images.load(img_name, check_existing=True) # bpy.data.images[img_name]
    img = np.array(img.pixels[:])
    return img


def save_image(fname, img):
    output_image = bpy.data.images.new('save_img', height=img.shape[0], width=img.shape[1])
    output_image.file_format = 'PNG'
    output_image.pixels = img.ravel()
    output_image.filepath_raw = fname
    output_image.save()


def overlay_smooth(img, render):
    img = np_array_from_image(img)
    render = np_array_from_image(render)
    img_size = int(np.sqrt(render.shape[0] // 4))

    render = render.reshape((img_size, img_size, 4))
    img = img.reshape((img_size, img_size, 4))

    # breakpoint()

    m = render[:, :, -1:] #  / 255.
    i = img[:, :, :3] * (1 - m) + render[:, :, :3] * m
    i = np.clip(i, 0., 1.) # .astype(np.uint8)
    i = np.concatenate([i, np.zeros((img_size, img_size, 1))], axis=-1)
    return i


def look_at(camera, point):
    loc_camera = camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    camera.rotation_euler = rot_quat.to_euler()
    return

def set_camera_translation(camera, x, y, z):
    # OpenCV applies translation to all vertices
    # In Blender we need to move camera in opposite direction to achieve same effect
    camera.location = (x, y, z)

    return

def get_vertices_global(obj):
    verts = obj.data.vertices
    vlen = len(verts)
    obj_matrix = np.matrix(obj.matrix_world.inverted().to_3x3())
    
    verts_co_1D = np.zeros([vlen*3], dtype='f')
    verts.foreach_get("co", verts_co_1D)
    verts_co_3D = verts_co_1D.reshape(vlen, 3)
    
    verts_co_3D = verts_co_3D @ obj_matrix
    verts_co_3D += np.array(obj.location)

    return verts_co_3D


def render_mesh(object_paths, output_dir, output_path, colors, wireframe, quads):
    ####################
    # Object
    ####################

    for i, (object_path, color) in enumerate(zip(object_paths, colors)):
        print("Loading obj: " + object_path)
        bpy.ops.wm.obj_import(filepath=object_path)
        object = bpy.context.selected_objects[0]

        mc = get_colors(color) / 255.

        bpy.data.materials['Body'].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (*mc, 1)

        # Check if the object has any material slots, if not, create one
        if len(object.data.materials) == 0:
            object.data.materials.append(None)

        object.data.materials[0] = bpy.data.materials['Body'].copy()

        if quads:
            bpy.context.view_layer.objects.active = object
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.tris_convert_to_quads()
            bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.shade_smooth()

    # Mark freestyle edges
    bpy.context.view_layer.objects.active = object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    camera = bpy.data.objects['Camera']

    scale = 1

    ####################
    # Render
    ####################
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.render.render(write_still=True)

    # Delete last selected object from scene
    object.select_set(True)
    bpy.ops.object.delete()

    return

##############################################################################
# Main
##############################################################################

if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    parser.add_argument('-i', '--inp', type=str, required=True, help='input directory')
    parser.add_argument('-o', '--out', type=str, default=None, help='output directory')
    parser.add_argument('-of', '--out_file', type=str, default=None, help='output file')
    parser.add_argument('-c', '--colors', type=str, default='turkuaz', help='mesh color')
    args = parser.parse_args()

    # argv = sys.argv
    # argv = argv[argv.index("--") + 1:]  # get all args after "--"

    print('Input arguments:', args)  # --> ['example', 'args', '123']

    if args.inp.endswith('.obj'):
        print('Processing a single file')
        input_file = args.inp
        input_file = os.path.abspath(input_file)
        filelist = [os.path.basename(args.inp)]
        input_dir = input_file.replace(os.path.basename(args.inp), '')
        output_dir = input_dir

    debug = False

    # Render setup
    scene = bpy.data.scenes['Scene']

    set_blender_gpu()

    # Change mesh color
    colors = args.colors.split('###')

    print('Num of files to be processed', len(filelist))

    mesh_fns = []
    for idx, input_file in enumerate(filelist):
        print(input_file)
        mesh_fn = os.path.join(input_dir, input_file)
        mesh_fns.append(mesh_fn)

    output_path = args.out_file
    render_mesh(
        mesh_fns,
        output_dir,
        output_path,
        colors=colors,
        wireframe=False,
        quads=True,
    )