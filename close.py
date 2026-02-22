import trimesh
from skimage import measure
from igl import signed_distance
import numpy as np
import os
import tqdm

# Resolution for the grid
reso = 256
# Create a 3D grid
grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T

# Root directory containing .obj files
root_dir = "datasets/gardata"

# List of all .obj files in the root directory
obj_files = [f for f in os.listdir(root_dir) if f.endswith('.obj')]

def compute_scale(mesh_path):
    # Load mesh or scene
    mesh = trimesh.load(mesh_path)

    # If it's a Scene, combine or extract meshes
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"No geometries found in {mesh_path}")
        # Combine all geometries into a single mesh
        mesh = trimesh.util.concatenate([g for g in mesh.dump()])

    # Normalize the mesh around its centroid
    mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
    return max(mesh.extents)  # Return the scale

# Full paths to .obj files
obj_file_paths = [os.path.join(root_dir, obj_file) for obj_file in obj_files]

# Compute the maximum scale across all meshes
all_scales = []
for path in tqdm.tqdm(obj_file_paths, desc="Computing scales"):
    try:
        all_scales.append(compute_scale(path))
    except Exception as e:
        print(f"Error processing {path}: {e}")
max_scale = max(all_scales)

# Process each .obj file
for obj_file, mesh_path in tqdm.tqdm(zip(obj_files, obj_file_paths), total=len(obj_files)):
    output_path = os.path.join('data/obj_meshes', f'{obj_file}')
    if os.path.exists(output_path):
        continue

    if not os.path.exists(mesh_path):
        print(f'skipping: {mesh_path}')
        continue

    try:
        # Load and scale the mesh
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                print(f"No geometries in scene: {mesh_path}")
                continue
            mesh = trimesh.util.concatenate([g for g in mesh.dump()])
        scene = trimesh.Scene(mesh)
        scene = scene.scaled(1.5 / max_scale)
        mesh = trimesh.util.concatenate([g for g in scene.dump()])

        mesh.rezero()
        mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)

        # Compute SDF and generate watertight mesh
        gt_sdf = signed_distance(grid, mesh.vertices, mesh.faces)[0][:, np.newaxis]
        gt_sdf = gt_sdf.reshape((reso, reso, reso))
        verts, faces, normals, values = measure.marching_cubes(gt_sdf, 0.00005)

        # Write the watertight mesh to an .obj file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as fp:
            for v in verts:
                fp.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for f in faces + 1:  # faces are 1-based, not 0-based in .obj files
                fp.write(f'f {f[0]} {f[1]} {f[2]}\n')
    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
