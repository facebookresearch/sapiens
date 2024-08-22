import pymeshlab
import os
from tqdm import tqdm
import numpy as np


def blender_vis(mesh_path, save_path, color='light_purple'):
    scene_file = './scene.blend'
    blender_file = 'render.py'
    thickness = 0

    output_file = save_path

    ####-------------------------------------------
    command = "blender -b {} \
        --python {} -- \
        -i {} \
        -o {} \
        -of {} \
        -c {}".format(\
                scene_file, blender_file, \
                mesh_path, mesh_path, output_file, color) 

    os.system(command)

    return


###-----------------------------------------------------
# colors = {
#         'pink': np.array([197, 27, 125]),
#         'light_pink': np.array([233, 163, 201]),
#         'light_green': np.array([161, 215, 106]),
#         'green': np.array([77, 146, 33]),
#         'red': np.array([215, 48, 39]),
#         'light_red': np.array([252, 146, 114]),
#         'light_orange': np.array([252, 141, 89]),
#         'purple': np.array([118, 42, 131]),
#         'light_purple': np.array([175, 141, 195]),
#         'light_blue': np.array([145, 191, 219]),
#         'blue': np.array([69, 117, 180]),
#         'gray': np.array([130, 130, 130]),
#         'white': np.array([255, 255, 255]),
#         'turkuaz': np.array([50, 134, 204]),
#         'yellow': np.array([235, 219, 52]),
#         # 'orange': np.array([242, 151, 43]),
#         'orange': np.array([220, 88, 42]),

#     }


mesh_dir = '/mnt/home/rawalk/drive/seg/data/iphone/point_clouds/pred'
color='medium_purple'

# # start_index = 4000; end_index = 4500;
# # start_index = 4500; end_index = 5000;
# # start_index = 5000; end_index = 5500;
# # start_index = 5500; end_index = 6000;
# # start_index = 6000; end_index = 6500;
# # start_index = 6500; end_index = 7000;
# start_index = 7000; end_index = -1;

# start_index = 4000; end_index = 5000;
# start_index = 5000; end_index = 6000;
# start_index = 6000; end_index = 7000;
start_index = 7000; end_index = -1;

# mesh_names = [
#     "004015.obj", "004019.obj", "004052.obj", "004054.obj", "004055.obj",
#     "004056.obj", "004057.obj", "004058.obj", "004059.obj", "004060.obj",
#     "004061.obj", "004062.obj", "004063.obj", "004064.obj", "004065.obj",
#     "004066.obj", "004067.obj", "004068.obj", "004069.obj", "004070.obj",
#     "004071.obj", "004072.obj", "004073.obj", "004074.obj", "004075.obj",
#     ]


mesh_names = sorted([x for x in os.listdir(mesh_dir) if x.endswith('.obj')])


##-------------------remaining mesh_names------------------------
present_image_names = sorted([x for x in os.listdir(mesh_dir + '_images') if x.endswith('.png')])
all_mesh_names = ['{:06d}.obj'.format(i) for i in range(0, 7283)]

mesh_names = [x for x in all_mesh_names if x.replace('.obj', '.png') not in present_image_names]

# start_index = 0; end_index = 100;
# start_index = 100; end_index = 200;
# start_index = 200; end_index = 300;
# start_index = 300; end_index = 400;
start_index = 400; end_index = -1;


##------------------------------------------------------------
if end_index == -1:
    end_index = len(mesh_names) - 1

output_mesh_dir = mesh_dir + '_images'
if not os.path.exists(output_mesh_dir):
    os.makedirs(output_mesh_dir)

mesh_names = mesh_names[start_index: end_index + 1]

# mesh_names = [
#     "003324.obj", "003325.obj", "003499.obj", "003787.obj", "003788.obj",
#     "003797.obj", "003800.obj", "003815.obj", "003816.obj", "003824.obj",
#     "003998.obj"
# ]

for i, mesh_name in enumerate(mesh_names):
    mesh_path = os.path.join(mesh_dir, mesh_name)
    save_path = os.path.join(output_mesh_dir, mesh_name.replace('.obj', '.png'))
    blender_vis(mesh_path, save_path, color)    
