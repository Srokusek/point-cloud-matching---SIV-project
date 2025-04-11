#
# this file contains all of the functions used throughout the project, including ones related to preprocessing, the models themselves and visualization
#

import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE

#
# ----------------------------------------------------functions used for preprocessing----------------------------------------------------
#

def prepare_point_cloud(pcd, voxel_size=0.05, show=False):
    """
    function for performing all 3 steps f preprocessing
        1. downsample
        2. estimate normals
        3. compute fpfh features
    """
    #1. downsample
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    #2. estimate normals
    radius_normal = voxel_size * 2
    down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    #3. compute FPFH features
    radius_feature = voxel_size*5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius = radius_feature,
            max_nn = 100
        )
    )

    return down_pcd, fpfh

#
# ----------------------------------------------------registration models----------------------------------------------------
#

def ransac_global_registration(source, target, voxel_size=0.05, icp = False, show=False, distance_threshold=0.05):
    """
    function performing the RANSAC global registration with possible local ICP registration
    """
    #prepare the point clouds
    s_down, s_fpfh = prepare_point_cloud(source, voxel_size=voxel_size)
    t_down, t_fpfh = prepare_point_cloud(target, voxel_size=voxel_size)

    #allow for visualization of the process
    if show:
        o3d.visualization.draw_geometries([s_down, t_down])    

    #perform the registration
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s_down, t_down,
        s_fpfh, t_fpfh,
        mutual_filter=False,
        max_correspondence_distance = distance_threshold
    )

    #perform further local registration if selected
    if icp:
        result_icp = o3d.pipelines.registration.registration_icp(
            s_down, t_down, distance_threshold*0.5, result_ransac.transformation
        )
        return result_icp
    else:
        return result_ransac
    
#function performing the fast global registration variation
def fast_global_registration(source, target, voxel_size=0.05, icp = False, show=False, distance_threshold=0.05):

    #prepare the point clouds
    s_down, s_fpfh = prepare_point_cloud(source, voxel_size=voxel_size)
    t_down, t_fpfh = prepare_point_cloud(target, voxel_size=voxel_size)

    if show:
        o3d.visualization.draw_geometries([s_down, t_down])

    #set the parameters
    fgr_option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance = distance_threshold
    )

    #perform the registration
    result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        s_down, t_down,
        s_fpfh, t_fpfh,
        fgr_option
    )

    #perform further local registration if selected
    if icp:
        result_icp = o3d.pipelines.registration.registration_icp(
            s_down, t_down, distance_threshold*0.5, result_fgr.transformation
        )
        return result_icp
    else:
        return result_fgr
    
#
# ----------------------------------------------------functions for running the experiments----------------------------------------------------
#

def test_high_res(source, thetas, registration_model, voxel_size=None, show=False):
    '''
    function to handle the iterative testing in the high resolution, no outliers experiment
    '''

    transformations = []
    scores = []

    #allow for visualizing of the process
    if show:
        o3d.visualization.draw_geometries([source])

    #test for each angle in the array of random angles
    for theta in thetas:
        s_pcd = copy.deepcopy(source)
        t_pcd = copy.deepcopy(source)

        #rotation matrix in x
        transformation_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), np.sin(theta), 0],
            [0, -np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

        #rotation matrix in y
        transformation_y = np.array([
            [np.cos(theta), 0, -np.sin(theta), 0],
            [0, 1, 0, 0],
            [np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

        #rotation matrix in z
        transformation_z = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        #apply the transformation to the scene point cloud
        s_pcd.transform(transformation_x)
        s_pcd.transform(transformation_y)
        s_pcd.transform(transformation_z)

        #center the clouds to origin to make the registration task simpler
        s_pcd.translate(-s_pcd.get_center())
        t_pcd.translate(-t_pcd.get_center())

        #allow for visualizing of the process
        if show:
            o3d.visualization.draw_geometries([s_pcd, t_pcd])

        #vary voxel size to be used for downsampling, if wanted
        #execute the registration
        if voxel_size:
            t0 = time.time()
            result = registration_model(s_pcd, t_pcd, voxel_size = voxel_size)
            elapsed = time.time() - t0
        else:
            t0 = time.time()
            result = registration_model(s_pcd, t_pcd)
            elapsed = time.time() - t0

        #apply the transformation found by the registration model
        s_pcd.transform(result.transformation)

        #save evalutation metrics
        score = {
            "Fitness": result.fitness,
            "RMSE": rmse(s_pcd, t_pcd),
            "Time": np.abs(elapsed)
        }
        scores.append(score)

        #save the transformations found
        transformations.append(result.transformation)

    return transformations, scores

def test_multiple_objects(sources, targets, registration_model, validation_pcd=None, voxel_size=0.05, distance_threshold=0.05):
    transformations = []
    scores = []
    t_pcds = []
    i = 0

    for target in targets:
        t_pcd = target.sample_points_poisson_disk(5000)
        t_pcds.append(t_pcd)

    for s_pcd in sources:

        for t_pcd in t_pcds:

            t0 = time.time()
            result = registration_model(s_pcd, t_pcd, voxel_size= voxel_size, distance_threshold=distance_threshold)
            elapsed = time.time()-t0

            if validation_pcd:
                s_temp = copy.deepcopy(validation_pcd[i])
            else:
                s_temp = copy.deepcopy(s_pcd)
            s_temp.transform(result.transformation)

            score = {
                "Fitness": result.fitness,
                "RMSE": rmse(s_temp, t_pcd),
                "Time": np.abs(elapsed)
            }
            scores.append(score)
            transformations.append(result.transformation)
        
            i += 1
    return transformations, scores

#
#----------------------------------------------------functions for evaluation and visualization----------------------------------------------------
#

def rmse(s_pcd, t_pcd):
    '''
    function for calculating the rmse between 2 pointclouds
    '''
    #use open3d library to calculate distances between pointclouds
    distances = s_pcd.compute_point_cloud_distance(t_pcd)
    
    #used the obtained distances to calculate mean and normalize using sqrt
    rmse = np.sqrt(np.mean(distances))
    return rmse



def visualize_fpfh_tsne(source_fpfh, target_fpfh):
    '''
    function to visualize fpfh features in 2d
    '''
    #transpose the features to (points x features) dimension
    source_features = np.asarray(source_fpfh.data).T
    target_features = np.asarray(target_fpfh.data).T

    #concat features to fit t-SNE on both target and source
    all_features = np.concatenate((source_features, target_features))

    #fit TSNE according to sklearn library, transform to project the features in 2d
    tsne = TSNE(n_components=2, random_state=44)
    features_2d = tsne.fit_transform(all_features)

    #split the source and target again to allow plotting with different colors
    source_2d = features_2d[:len(source_features)]
    target_2d = features_2d[len(source_features):]

    #plot the points in the reduced dim, differentiate source and target using colors
    plt.figure(figsize=(10, 6))
    plt.scatter(source_2d[:, 0], source_2d[:, 1], label="Soure Features", alpha=0.7, color=[1, 0.7, 0])
    plt.scatter(target_2d[:, 0], target_2d[:, 1], label="Target Features", alpha=0.7, color=[0, 0.6, 0.9])
    plt.title("t-SNE Projection of FPFH Features")
    plt.legend()
    plt.show()


def draw_registration_result(source, target, transformation, pov_params=None):
    '''
    function meant to show the result of the transformation found by any registration method
    '''
    #make copies of source and target to not alter original pointcloud
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    #paint clouds for better visualization
    source_temp.paint_uniform_color([1, 0.7, 0])
    target_temp.paint_uniform_color([0, 0.6, 0.9])
    
    #apply the transformation to be visualized
    source_temp.transform(transformation)

    #visualize
    if pov_params: #if a specific view is defined, visualize from this view point
        o3d.visualization.draw_geometries([source_temp, target_temp], **pov_params)
    else:
        o3d.visualization.draw_geometries([source_temp, target_temp])


def plot_fpfh(fpfh, title="FPFH Features"):
    '''
    Function for plotting the fpfh values
    '''
    fpfh_data = np.asarray(fpfh.data)
    plt.figure(figsize=(10, 4))
    plt.plot(fpfh_data.mean(axis=1))
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Value")
    plt.show()

def compare_ransac_vs_fgr(scores_ransac, scores_fgr, metric="Fitness"):
    '''
    Function for making comparisons between the registration models on various metrics
    '''
    #extract the metric values for each trial
    ransac_values = [score[metric] for score in scores_ransac]
    fgr_values = [score[metric] for score in scores_fgr]

    #calculate the mean values
    mean_ransac = np.mean(ransac_values)
    mean_fgr = np.mean(fgr_values)

    #create an array of trial indices
    trials = np.arange(len(ransac_values))

    #set up the plot
    width = 0.4 
    fig, ax = plt.subplots(figsize=(16, 8))  

    #plot RANSAC and FGR values side by side
    ax.bar(trials - width / 2, ransac_values, width, label="RANSAC", color=[1, 0.7, 0], edgecolor="black")
    ax.bar(trials + width / 2, fgr_values, width, label="FGR", color=[0, 0.6, 0.9], edgecolor="black")

    #add mean lines
    ax.axhline(mean_ransac, color="orange", linestyle="--", linewidth=2, label=f"RANSAC Mean ({mean_ransac:.2f})")
    ax.axhline(mean_fgr, color="blue", linestyle="--", linewidth=2, label=f"FGR Mean ({mean_fgr:.2f})")

    #add labels, title, and legend
    ax.set_xlabel("Trial", fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(f"Comparison of {metric} for RANSAC vs FGR", fontsize=18, fontweight="bold")
    ax.set_xticks(trials[::10])  # Show every 10th trial on the x-axis for clarity
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=12)

    #add gridlines for better readability
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    #annotate the bars with their values (only for a subset to avoid clutter)
    if len(trials) <= 20:  #annotate only if there are few trials
        for i, value in enumerate(ransac_values):
            ax.text(i - width / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
        for i, value in enumerate(fgr_values):
            ax.text(i + width / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=10)

    #adjust layout for better spacing
    plt.tight_layout()
    plt.show()


''' 
#Prior experimentations with the opencv library
#Decided it is easier to work with open3d instead, as it focuses on 3d data
#and has better pointcloud functionalities


def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl

def ransac(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    plane, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    return inliers

def preprocess_pointcloud(pcd, voxel_size=0.1, remove_planes=True):
    #pcd.translate(-pcd.get_center())
    pcd_clean = remove_statistical_outliers(pcd)
    if remove_planes:
        indices = ransac(pcd_clean)
        pcd_clean = pcd_clean.select_by_index(indices, invert=True)
    downpcd = pcd_clean.voxel_down_sample(voxel_size=voxel_size)
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return downpcd

def o3d_to_cv(pcd):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return np.hstack((points, normals)).astype(np.float32)

def load_and_preprocess(model_path, scene_path, num_points=2000):
    """Load and preprocess the model and scene"""
    #Load model
    mesh = o3d.io.read_triangle_mesh(model_path)
    model_pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    #Load scene
    scene_pcd = o3d.io.read_point_cloud(scene_path)
    
    #Basic preprocessing for both point clouds
    model_pcd = preprocess_pointcloud(model_pcd, voxel_size=0.05, remove_planes=False)
    scene_pcd = preprocess_pointcloud(scene_pcd, voxel_size=0.05, remove_planes=False)
    
    return model_pcd, scene_pcd

def match_model_to_scene(model_pcd, scene_pcd, num_results=2):
    """Match model to scene using PPF and ICP"""
    #Convert to OpenCV format
    model_formatted = o3d_to_cv(model_pcd)
    scene_formatted = o3d_to_cv(scene_pcd)
    
    #Train detector with adjusted parameters
    detector = cv.ppf_match_3d_PPF3DDetector(0.05, 0.05)
    print("Training model...")
    detector.trainModel(model_formatted)
    
    #Match with adjusted parameters
    print("Matching...")
    results = detector.match(scene_formatted, 0.05, 0.05)
    
    if len(results) == 0:
        print("No matches found")
        return [], None
    
    print(f"Found {len(results)} initial matches")
    
    #Refine with ICP
    icp = cv.ppf_match_3d_ICP(100)  # 100 iterations
    print("Refining with ICP...")
    _, results = icp.registerModelToScene(model_formatted, 
                                        scene_formatted, 
                                        results[:num_results])
    
    return results, model_formatted

def find_matching_cluster(scene_pcd, transformed_pcd, eps=0.5, min_points=10):
    #Separate clustering function
    labels = np.array(scene_pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return int(np.median(labels)) if len(labels) > 0 else -1

def main(model_path, scene_path, output_path):
    print(f"Processing model: {model_path}")
    print(f"Scene: {scene_path}")
    
    #Load and preprocess
    model_pcd, scene_pcd = load_and_preprocess(model_path, scene_path)
    
    #Match model to scene
    results, model_formatted = match_model_to_scene(model_pcd, scene_pcd)
    
    #Transform and save best match
    if len(results) > 0:
        best_result = results[0]
        output_file = f"{output_path}/transformed_model.ply"
        
        transformed_points = cv.ppf_match_3d.transformPCPose(
            model_formatted, best_result.pose)
        cv.ppf_match_3d.writePLY(transformed_points, output_file)
        
        print(f"Best match: {best_result.numVotes} votes, residual: {best_result.residual}")
        
        #Find matching cluster
        transformed_pcd = o3d.io.read_point_cloud(output_file)
        #matching_cluster, _ = match_model_to_scene(scene_pcd, transformed_pcd)
        
        return {
            'pose': best_result.pose,
            'num_votes': best_result.numVotes,
            'residual': best_result.residual,
            'matching_cluster': find_matching_cluster(scene_pcd, transformed_pcd)
        }
    
    print("No matches found")
    return None
'''