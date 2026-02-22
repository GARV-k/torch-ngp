import os

# Specify the folder containing the .obj files
t = '/workspace/LLM-3D/torch-ngp-mini2/data/obj_meshes'
for folder_path in [t,t+'_90',t+'_180',t+'_270']:
    import os


    # List all files ending with .obj in the folder
    obj_files = [f for f in os.listdir(folder_path) if f.endswith('.obj')]

    # Sort files to maintain a consistent order
    obj_files.sort()

    # Rename files sequentially as 0.obj, 1.obj, ..., N.obj
    temp_names = []

    # Step 1: Temporarily rename files to avoid conflicts
    for index, obj_file in enumerate(obj_files):
        temp_name = f"temp_{index}.obj"
        temp_names.append(temp_name)
        os.rename(os.path.join(folder_path, obj_file), os.path.join(folder_path, temp_name))

    # Step 2: Rename files sequentially
    for new_index, temp_name in enumerate(temp_names):
        new_name = f"{new_index}.obj"
        os.rename(os.path.join(folder_path, temp_name), os.path.join(folder_path, new_name))
        print(f"Renamed: {temp_name} -> {new_name}")

    print("Renaming complete!")


