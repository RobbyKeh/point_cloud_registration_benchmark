import os
import time
import numpy as np
from numpy.core.fromnumeric import transpose
import open3d as o3d 
import matplotlib
import copy

#######################################################
###### Point Cloud and Solution Helper Functions ######
#######################################################

def loadPointCloud(pcloud):
    
    # pcloud = o3d.io.read_point_cloud(pcd_file)
    # pcloud = np.asarray(pcloud.points) # Load in source point cloud

    plist = pcloud.tolist()
    p3dlist = []
    for x,y,z in plist:
        pt = POINT3D(x,y,z)
        p3dlist.append(pt)
    return pcloud.shape[0], p3dlist, pcloud

def calculate_error(cloud1: o3d.geometry.PointCloud, cloud2: o3d.geometry.PointCloud) -> float:
    # Compare the aligned source point cloud with the original source point cloud
    # 'metric.py' from 'metric' folder 
    # assert len(cloud1.points) != len(cloud2.points), "len(cloud1.points) != len(cloud2.points)"
    
    centroid, _ = cloud1.compute_mean_and_covariance()
    weights = np.linalg.norm(np.asarray(cloud1.points) - centroid, 2, axis=1)
    distances = np.linalg.norm(np.asarray(cloud1.points) - np.asarray(cloud2.points), 2, axis=1)/len(weights)
    return np.sum(distances/weights)

def readResultsFile(datatext):
    # This function reads a single line of the inputted *_global.txt file containing the registration problem to solve
    # 
    # This function extracts the following information:
        # id A unique identifier of the registration problem;
        # source name The file name of the source point cloud;
        # target name The file name of the target point cloud;
        # overlap The percentage of overlap between the two point clouds;
        # t1..t12 The elements of the 4x4 transformation matrix representing the initial misplacement to apply. 
        #         The last line is implicit, since for a rototranslation is always the same; therefore, the matrix is
    
    # Must be in a dataset folder directory, one of these ['eth', 'kaist', 'planetary', 'tum']
    with open(datatext) as f:
        lines = f.readlines()
        results_arr = [] # Create empty list for storing results
        for i in range(1, len(lines)): # do not need header information, start at index 1
            line = lines[i].split()
            results_arr.append(line) # reads the entire file
    for row in range(0, len(results_arr)):
        for col in range(0,9):
            if col != 1 or col != 2: 
                results_arr[row][col] = float(results_arr[row][col])
            else:
                results_arr[row][col] = results_arr[row][col]

    return results_arr

def draw_registration_result(source, target, transformation):
    # Helper Visualization function
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 0, 1]) # show source in blue
    target_temp.paint_uniform_color([1, 0, 0]) # show target in red
    source_temp.transform(transformation) # Transform source
    o3d.visualization.draw_geometries([source_temp, target_temp]) # plot A and B 

def getRotMatAndTransVec(affineTrans):
    # Get the 3x3 rotation matrix and 3x1 translation vector from the 4x4 affine transformation matrix
    # Input: transformation - the 4x4 affine transformation matrix
    rot_mat = np.array([[affineTrans[0][0],  affineTrans[0][1],  affineTrans[0][2]], # Rotation Matrix
                        [affineTrans[1][0],  affineTrans[1][1],  affineTrans[1][2]],
                        [affineTrans[2][0], affineTrans[2][1], affineTrans[2][2]]]) 
    translation = np.transpose(np.array([affineTrans[0][3], affineTrans[1][3], affineTrans[2][3]])) # Translation Matrix
    return rot_mat, translation

def convert2affineTrans(rot_mat, translation):
    # Convert inputted rotation matrix and translation vector to a 4x4 affine transformation matrix or 'rototranslation' matrix
    affineTrans = np.zeros((4,4))
    affineTrans[:3, :3] = rot_mat
    affineTrans[:,3] = np.append(translation, 1)
    return affineTrans

def xyz2pcd(xyz_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_array)
    return pcd

def pcd2xyz(pcd):
    xyz_array = np.asarray(pcd.points) 
    return xyz_array

def transposeAffine(affineTrans):
    rot_mat, translation = getRotMatAndTransVec(affineTrans)
    affineTrans_transposed = convert2affineTrans(np.transpose(rot_mat), translation)
    return affineTrans_transposed

def printError(true_rot_mat, true_translation, solution_transformation):
    sol_rot_mat, sol_trans_vec = getRotMatAndTransVec(solution_transformation)
    diff_rot_mat = np.matmul(true_rot_mat, np.transpose(sol_rot_mat))
    diff_trans_vec = true_translation - sol_trans_vec
    print("Rot Mat Difference")
    print(diff_rot_mat)
    print("Translation Vector Difference")
    print(diff_trans_vec)

def getError(true_rot_mat, true_translation, solution_transformation):
    sol_rot_mat, sol_trans_vec = getRotMatAndTransVec(solution_transformation)
    diff_rot_mat = np.matmul(true_rot_mat, np.transpose(sol_rot_mat))
    diff_trans_vec = true_translation - sol_trans_vec
    return diff_rot_mat, diff_trans_vec

def get_angular_error(R_exp, R_est):
    """
    Calculate angular error between experimental and true rotation
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)))

def convertArraysToSameDimension(src_pc, tgt_pc): # do not use
    src_len = len(src_pc)
    tgt_len = len(tgt_pc)
    
    beg = time.time()
    print("Size of Source Point Cloud BEFORE random point deletion: %d" %src_len)
    print("Size of Target Point Cloud BEFORE random point deletion: %d" %tgt_len)
    print("Deleting random points to get equal dimensions of source and target point clouds....")
    while src_len != tgt_len:
        src_len = len(src_pc)
        tgt_len = len(tgt_pc)
        if src_len > tgt_len:
            rand_ndx = np.random.randint(0, src_len)
            src_pc = np.delete(src_pc, rand_ndx, axis=0) # Delete random row
        else:
            rand_ndx = np.random.randint(0, tgt_len)
            tgt_pc = np.delete(tgt_pc, rand_ndx, axis=0) # Delete random row
    print("This took %0.4f seconds:" %(time.time() - beg))
    print("Size of Source Point Cloud AFTER random point deletion: %d" %len(src_pc))
    print("Size of Target Point Cloud AFTER random point deletion: %d" %len(tgt_pc))

    return src_pc, tgt_pc

# We downsample the point cloud, estimate normals, then compute a FPFH feature for each point. 
# The FPFH feature is a 3-dimensional vector that describes the local geometric property of a point. 
# A nearest neighbor query in the 3-dimensional space can return points with similar local geometric structures. 
def preprocess_point_cloud(pcd, voxel_size):
    # See http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Extract-geometric-feature 
    # Source: Rusu, N. Blodow, and M. Beetz, Fast Point Feature Histograms (FPFH) for 3D registration, ICRA, 2009.
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def voxel_downsampling(source, target, voxel_size):
    ## Voxel Downsampling - See http://www.open3d.org/docs/0.7.0/tutorial/Basic/pointcloud.html
    
    # Inputs:
    #   source - source point cloud to be transformed as a .pcd or point cloud data type
    #   target - target point cloud to be transformed as a .pcd or point cloud data type
    #   voxel_size_num - voxel size to downsample

    # To increase registration speed, we perform voxel downsampling and visualize the downsampled point clouds.
    print("Downsampling the point cloud using voxel downsmapling...")
    print("Size of Source Point Cloud BEFORE downsampling: %d" %len(source.points))
    print("Size of Target Point Cloud BEFORE downsampling: %d" %len(target.points))
    # o3d.visualization.draw_geometries([source, target])
    source_down = o3d.geometry.PointCloud.voxel_down_sample(source, voxel_size)
    target_down = o3d.geometry.PointCloud.voxel_down_sample(target, voxel_size)
    print("Size of Source Point Cloud AFTER voxel downsampling: %d" %len(source_down.points))
    print("Size of Target Point Cloud AFTER voxel downsampling: %d" %len(target_down.points))
    # o3d.visualization.draw_geometries([source_down, target_down])
    return source_down, target_down

def uniform_downsampling(source, target, k_points):
    
    ## Uniform Downsampling

    # Inputs:
    #   source - source point cloud to be transformed as a .pcd or point cloud data type
    #   target - target point cloud to be transformed as a .pcd or point cloud data type
    #   k_points - number of points size to uniformly downsample

    ## Uniform Downsampling - downsample point cloud points uniformly for every "k" points
    print("Downsampling the point cloud using voxel downsmapling...")
    print("Size of Source Point Cloud BEFORE downsampling: %d" %len(source.points))
    print("Size of Target Point Cloud BEFORE downsampling: %d" %len(target.points))
    source_down = o3d.geometry.PointCloud.uniform_down_sample(source, k_points)
    target_down = o3d.geometry.PointCloud.uniform_down_sample(target, k_points)
    print("Size of Source Point Cloud AFTER voxel downsampling: %d" %len(source_down.points))
    print("Size of Target Point Cloud AFTER voxel downsampling: %d" %len(target_down.points))

def createDictEntry(dataset_folders_name, datatext_name):
    # Create preinitialized dictionary for storing results based on algorithm name
    entry_dict = {'dataset_folder_name': dataset_folders_name,
                    'datatext': datatext_name,
                    'id_str': [],
                    'src_str': [],
                    'tgt_str': [],
                    'overlap': [],
                    'alg_time': [],
                    'ang_err': [],
                    'err_met': [],
                    'trans_err_vec': []}
    return entry_dict

class results:
    def __init__(self, alg_name) -> None:
        algorithm = alg_name
        # dataset_folder_name = dataset_folders_name,
        # datatext = datatext_name
        # id_str =
        # src_str =
        # tgt_str =
        # overlap =
        # alg_time =
        # ang_err =
        # err_met' = 
        # trans_err_vec = 

    
def getResults(algorithm):

    # Change to "RESULTS" folder
    os.chdir("./RESULTS")

    # Create empty dictionary for storing each algorithm's results
    results_dict = {'alg_name': algorithm}

    # Specify which folders contain datasets to be analyzed
    algorithm_results_folders = sorted([file for file in os.listdir(".") if file.endswith('_RESULTS')]) # for GLOBAL registration algorithms

    # Loop over results folders
    for i in range(0, len(algorithm_results_folders)):

        # Change current directory to each dataset folder
        os.chdir(algorithm_results_folders[i])
        print('Changing dataset directory to: ' + algorithm_results_folders[i])
        # In each dataset folder, find all dataset folders found within each algorithm results folder e.g. 'apartment' in 'eth' dataset folder
        dataset_folders = sorted([folder_name for folder_name in os.listdir(".") if os.path.isdir(folder_name)]) 
        
        for j in range(0, len(dataset_folders)):
            os.chdir(dataset_folders[j])
            datatext = sorted([file for file in os.listdir(".") if file.endswith('.txt')])

            # Iterate through datatext list to read registration problems from _global.txt files
            for k in range(0, len(datatext)):

                # # Get the registration problem
                # registration_problem = readResultsFile(results_file) # N by 8 multidimensional list 

                # Read results text file for each dataset folder
                with open(datatext[k], 'r') as results_file:
                    results_arr = []
                    lines = results_file.readlines()
                    for i in range(1, len(lines)): # do not need header information, start at index 1
                        line = lines[i].split()
                        results_arr.append(line)

                print('------> Analyzing results file: ' + datatext[k])

                results_dict_temp = createDictEntry(dataset_folders[j], datatext[k])

                # Now read in raw results data as specified for each registration problem
                for row in range(0, len(results_arr)):
                    results_dict_temp['id_str'].append(results_arr[row][0])  # Unique identifier of the registration problem
                    results_dict_temp['src_str'].append(results_arr[row][1]) # File name of the Source Point Cloud
                    results_dict_temp['tgt_str'].append(results_arr[row][2]) # File name of the Target Point Cloud
                    results_dict_temp['overlap'].append(float(results_arr[row][3])) # Percentage of overlap between the two point clouds
                    results_dict_temp['alg_time'].append(float(results_arr[row][4]))  # Total time taken by algorithm to find the registation solution for each registration problem
                    results_dict_temp['ang_err'].append(float(results_arr[row][5])) # Angular error between the true and experimental transformation aligning the source to the target point cloud
                    results_dict_temp['err_met'].append(float(results_arr[row][6])) # Compare the aligned source point cloud with the original source point cloud
                    results_dict_temp['trans_err_vec'].append([float(results_arr[row][7]),  float(results_arr[row][8]),  float(results_arr[row][9])])  # Get translation error

            os.chdir("..") # Change directory back to algorithm dataset folder, e.g. 'Go-ICP_RESULTS'
    
        # Go back to parent directory "RESULTS" folder and get ready to initialize next algorithm results folder
        os.chdir('..')

    return results_dict

if __name__ == '__main__':
    
    # Time the  project
    start_time = time.time()

    # Setup initial parameters
    local_registration_algs = ["ICP_point2point", "ICP_point2plane"] # local registration algorithms
    global_registration_algs = ["FGR", "Go-ICP", "TEASER", "RANSAC"] # global registration algorithms
    all_algorithms = local_registration_algs + global_registration_algs

    # For simplicity purposes, only test a single algorithm at a time
    # algorithm = local_registration_algs[0]
    # algorithm = global_registration_algs[2]
    algorithm = all_algorithms[0]

    # Call main function
    results_dict = getResults(algorithm)

    total_time = time.time() - start_time
    print('Results DONE')
    print('Total time completed in ' + str(total_time) + ' seconds')

