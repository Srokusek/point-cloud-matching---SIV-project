#
# this file is meant to create a dataset using blender. It uses simple objects in a randomized grid along with the Blainder addon to generate the point clouds.
# furthermore, the correct object from the scene is scanned separately to allow for better evaluation
#

import bpy
import bmesh
import random
import range_scanner
import time


def test_model(iterations, cam="horizontal", seed=44):

    #create object meshes
    def create_cube():
        mesh = bpy.data.meshes.new("CubeMesh")
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1)
        bm.to_mesh(mesh)
        bm.free()

        return mesh

    def create_sphere():
        mesh = bpy.data.meshes.new("SphereMesh")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.5)
        bm.to_mesh(mesh)
        bm.free()

        return mesh

    def create_cylinder():
        mesh = bpy.data.meshes.new("CylinderMesh")
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.5, radius2=0.5, depth=1)
        bm.to_mesh(mesh)
        bm.free()

        return mesh
    
    #create basic material (necessary for the scanner addon to work properly)
    material = bpy.data.materials.new(name="BasicMaterial")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)

    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    #add camera
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)

    #set camera as active camera
    bpy.context.scene.camera = camera_object

    if cam == "horizontal":
        #animate camera movement
        positions = [(13, -5, 8), (13, 0, 8), (13, 5, 8)]
        rotations = [(1.1, 0, 1.256), (1.1, 0, 1.57), (1.1, 0, 1.884)]

        frame_start = 1
        frames_per_position = 5
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = frame_start + frames_per_position * (len(positions) - 1)

    #create keyframes of camera animation
    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        frame = frame_start + (i * frames_per_position)

        camera_object.location = pos
        camera_object.keyframe_insert(data_path="location", frame=frame)

        camera_object.rotation_euler = rot
        camera_object.keyframe_insert(data_path="rotation_euler", frame=frame)

    for i in range(iterations):
        #make new seed for each iteration
        random.seed(seed + 33*i)

        #clear the init scene
        for obj in bpy.data.objects:
            if obj.type != "CAMERA":
                obj.select_set(True)
            else:
                obj.select_set(False)
        bpy.ops.object.delete(use_global=False)


        #list of possible meshes
        mesh_creators = [create_cube, create_cylinder, create_sphere]
        meshes = [creator() for creator in mesh_creators]

        #create position grid for testing
        grid_size = 5
        num_objects = len(meshes)

        grid_positions = [(-2 * i, 2 * j - 3, 0) for i in range(grid_size) for j in range(grid_size)]

        #randomly select locations
        selected_positions = random.sample(grid_positions, num_objects)

        for mesh, pos in zip(meshes, selected_positions):
            obj = bpy.data.objects.new(f"{mesh.name}Object", mesh)
            obj.location = pos
            obj.data.materials.append(material)
            bpy.context.collection.objects.link(obj)

        #set the active object to the camera
        bpy.context.view_layer.objects.active = bpy.context.scene.camera

        #set correct frame
        bpy.context.scene.frame_set(1)


        #set up Blainder addon for simulating scanning
        range_scanner.ui.user_interface.scan_static(
            bpy.context, 

            scannerObject=bpy.context.scene.objects["Camera"],

            resolutionX=600, fovX=60, resolutionY=600, fovY=60, resolutionPercentage=100,

            reflectivityLower=0.0, distanceLower=0.0, reflectivityUpper=0.0, distanceUpper=99999.9, maxReflectionDepth=10,
            
            enableAnimation=True, frameStart=1, frameEnd=frame_start + frames_per_position * (len(positions) - 1), frameStep=1, frameRate=1,

            addNoise=False, noiseType='gaussian', mu=0.0, sigma=0.01, noiseAbsoluteOffset=0.0, noiseRelativeOffset=0.0,

            simulateRain=False, rainfallRate=0.0, 

            addMesh=False,

            exportLAS=False, exportHDF=False, exportCSV=False, exportPLY=True, exportSingleFrames=False,
            exportRenderedImage=False, exportSegmentedImage=False, exportPascalVoc=False, exportDepthmap=False, depthMinDistance=0.0, depthMaxDistance=100.0, 
            dataFilePath="/home/simon/repos/signal-processing/project/dataset/test_x", dataFileName=f"iteration_{i}",
            
            debugLines=False, debugOutput=False, outputProgress=True, measureTime=False, singleRay=False, destinationObject=None, targetObject=None
        )    

        #scan for the validation
        #remove non-target objects and keep only the correct object

        for obj in bpy.data.objects:
            if "SphereMesh" in obj.name or "CylinderMesh" in obj.name:
                bpy.data.objects.remove(obj, do_unlink=True)

        bpy.context.view_layer.update()

        range_scanner.ui.user_interface.scan_static(
            bpy.context, 

            scannerObject=bpy.context.scene.objects["Camera"],

            resolutionX=600, fovX=60, resolutionY=600, fovY=60, resolutionPercentage=100,

            reflectivityLower=0.0, distanceLower=0.0, reflectivityUpper=0.0, distanceUpper=99999.9, maxReflectionDepth=10,
            
            enableAnimation=True, frameStart=1, frameEnd=frame_start + frames_per_position * (len(positions) - 1), frameStep=1, frameRate=1,

            addNoise=False, noiseType='gaussian', mu=0.0, sigma=0.01, noiseAbsoluteOffset=0.0, noiseRelativeOffset=0.0,

            simulateRain=False, rainfallRate=0.0, 

            addMesh=False,

            exportLAS=False, exportHDF=False, exportCSV=False, exportPLY=True, exportSingleFrames=False,
            exportRenderedImage=False, exportSegmentedImage=False, exportPascalVoc=False, exportDepthmap=False, depthMinDistance=0.0, depthMaxDistance=100.0, 
            dataFilePath="/home/simon/repos/signal-processing/project/dataset/test_y", dataFileName=f"iteration_{i}",
            
            debugLines=False, debugOutput=False, outputProgress=True, measureTime=False, singleRay=False, destinationObject=None, targetObject=None
        )


#create 100 samples
test_model(iterations=100)