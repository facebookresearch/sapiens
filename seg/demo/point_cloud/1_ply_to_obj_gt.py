import pymeshlab
import os
from tqdm import tqdm
import numpy as np

def save_obj_with_vertex_colors(mesh, filename):
    with open(filename, 'w') as f:
        # Write vertices with colors
        for v, c in zip(mesh.vertex_matrix(), mesh.vertex_color_matrix()):
            f.write(f'v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n')

        # Write faces
        for face in mesh.face_matrix():
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def center_mesh_and_scale(ms, centroid=None, scale_factor=0.01):
    mesh = ms.current_mesh()

    if centroid is None:
        vertices = mesh.vertex_matrix()
        centroid = vertices.mean(axis=0)

    translation_vector = -centroid
    # Rotate 180 degrees around the X-axis to flip the mesh
    ms.apply_filter('compute_matrix_from_translation_rotation_scale', translationx=translation_vector[0], translationy=translation_vector[1], translationz=translation_vector[2])
    ms.apply_filter('compute_matrix_from_translation_rotation_scale', rotationx=180,)
    ms.apply_filter('compute_matrix_from_translation_rotation_scale', scalex=scale_factor, scaley=scale_factor, scalez=scale_factor)
    
    return centroid

##---------------convert .ply to .obj-------------------
point_cloud_dir = '/mnt/home/rawalk/drive/seg/data/iphone/point_clouds/gt'
point_cloud_names = sorted([x for x in os.listdir(point_cloud_dir) if x.endswith('.ply')])
# start_index = 0; end_index = 200;
# start_index = 200; end_index = 400;
# start_index = 400; end_index = 600;
# start_index = 600; end_index = 800;
# start_index = 800; end_index = 1000;
# start_index = 1000; end_index = 1200;
# start_index = 1200; end_index = 1400;
# start_index = 1400; end_index = 1600;
# start_index = 1600; end_index = 1800;
# start_index = 1800; end_index = 2000;

# start_index = 2000; end_index = 2200;
# start_index = 2200; end_index = 2400;
# start_index = 2400; end_index = 2600;
# start_index = 2600; end_index = 2800;
# start_index = 2800; end_index = 3000;
# start_index = 3000; end_index = 3200;
# start_index = 3200; end_index = 3400;
# start_index = 3400; end_index = 3600;
# start_index = 3600; end_index = 3800;
# start_index = 3800; end_index = 4000;

# start_index = 4000; end_index = 4200;
# start_index = 4200; end_index = 4400;
# start_index = 4400; end_index = 4600;
# start_index = 4600; end_index = 4800;
# start_index = 4800; end_index = 5000;
# start_index = 5000; end_index = 5200;
# start_index = 5200; end_index = 5400;
# start_index = 5400; end_index = 5600;
# start_index = 5600; end_index = 5800;
# start_index = 5800; end_index = 6000;
# start_index = 6000; end_index = 6200;
# start_index = 6200; end_index = 6400;
# start_index = 6400; end_index = 6600;
# start_index = 6600; end_index = 6800;
# start_index = 6800; end_index = 7000;
# start_index = 7000; end_index = 7200;
start_index = 7200; end_index = -1;

if end_index == -1:
    end_index = len(point_cloud_names) - 1

point_cloud_names = point_cloud_names[start_index: end_index + 1]

centroid = np.array([10.8753141, 266.63371396, 649.56045929]) ## computed from the centroid of the first pred frame of the video
scale_factor = 0.01

for point_cloud_name in tqdm(point_cloud_names):
    point_cloud_path = os.path.join(point_cloud_dir, point_cloud_name)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(point_cloud_path)

    # Center the mesh at the origin and scale it down
    centroid = center_mesh_and_scale(ms, centroid=centroid, scale_factor=scale_factor)

    ms.generate_surface_reconstruction_ball_pivoting(clustering=0.01)

    # Get the current mesh
    mesh = ms.current_mesh()
    save_obj_with_vertex_colors(mesh, point_cloud_path.replace('.ply', '.obj'))

