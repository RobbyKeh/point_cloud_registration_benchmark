import os
import time
import numpy as np
from numpy.core.fromnumeric import transpose
import open3d as o3d 
import teaserpp_python
import matplotlib
import copy
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE;

# transform: Coordinate transformation of a point cloud
# https://www.geo.tuwien.ac.at/downloads/pg/pctools/publish/pointCloudHelp/transform.html
#   DESCRIPTION/NOTES
#   * Transformation model:
#     ----------------------------
#      xNew = m * A * xActual + t
#     ----------------------------
#     where m ... 1-by-1 scale
#           A ... 3-by-3 matrix
#           t ... 3-by-1 translation vector -> [tx; ty; tz]
 
#    * If coordinates should only be scaled, use the identity matrix as A (command
#      'eye(3)') and a null vector as translation vector (command 'zeros(3,1)').
 
#    * If present, also the normal vectors are transformed.
#   ------------------------------------------------------------------------------
#   INPUT
#   1 [m]
#     Scale as scalar.
  
#   2 [A]
#     Any matrix of size 3-by-3.
 
#   3 [t]
#     Translation vector of size 3-by-1.
#   ------------------------------------------------------------------------------
#   OUTPUT
#   1 [obj]
#     Updated object.
#   ------------------------------------------------------------------------------
#   EXAMPLES
#   1 Rotate point cloud by 100 gradians (=90 degree) about z axis.
#     pc = pointCloud('Lion.xyz');
#     R = opk2R(0, 0, 100); % create rotation matrix
#     pc.transform(1, R, zeros(3,1)); % no scale, no translation
#     pc.plot;
 
#   2 Apply only scale.
#     pc = pointCloud('Lion.xyz');
#     pc.transform(1e-3, eye(3), zeros(3,1)); % transformation from mm -> m
#     pc.plot;
#   ------------------------------------------------------------------------------
#   philipp.glira@gmail.com
#   ------------------------------------------------------------------------------

# Open3D also supports a general transformation defined by a 4×4 homogeneous
# transformation matrix using the method transform.

#############################################
##### Point Set Registration Algorithms #####
#############################################

def ICP_point2point(source, target, threshold, trans_init): 
# Open3D Point-to-Point ICP registration
# See tutorials here -> http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
# Input point clouds must be in .pcd format

    print("Apply point-to-point ICP")
    max_iter = 30 # default
    start = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    total_time = time.time() - start
    print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)

    return reg_p2p, total_time

def ICP_point2plane(source, target, threshold, trans_init): # Input point clouds must be in .pcd format
# Open3D Point-to-Plane ICP registration - Ideally, this method will be faster than Point-to-Point ICP
# See tutorials here -> http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html

    print("Apply point-to-plane ICP")

    start = time.time()
    # Compute Point Normal Estimations
    print("Computing Point Normal Estimations...")
    o3d.geometry.PointCloud.estimate_normals(source,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.geometry.PointCloud.estimate_normals(target,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    total_time = time.time() - start
    print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # draw_registration_result(source, target, reg_p2l.transformation)
    return reg_p2l, total_time

def goICP(target, source): 
    goicp = GoICP()
    goicp.MSEThresh = 0.001;
    goicp.trimFraction = 0.0;
    
    if(goicp.trimFraction < 0.001):
        goicp.doTrim = False;

    Nm, a_points, np_a_points = loadPointCloud(target)
    Nd, b_points, np_b_points = loadPointCloud(source)
    goicp.loadModelAndData(Nm, a_points, Nd, b_points)

    #LESS DT Size = LESS TIME CONSUMPTION = HIGHER ERROR
    goicp.setDTSizeAndFactor(100, 1.0)

    start = time.time()
    print("Building Distance Transform...")
    goicp.BuildDT()
    print("Done with BDT")
    print("REGISTERING....")
    goicp.Register()
    print("Done with Registering")
    total_time = time.time() - start
    optR = np.array(goicp.optimalRotation()) # A python list of 3x3 is returned with the optimal rotation
    optT = np.array(goicp.optimalTranslation()) # A python list of 1x3 is returned with the optimal translation
    T_source2target = convert2affineTrans(optR, optT)
    
    return T_source2target, total_time

def execute_RANSAC_global_registration(source, target, voxel_size):
# For documentation, see http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#RANSAC

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    start = time.time()
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    total_time = time.time() - start

    return result, total_time

def execute_fast_global_registration(source, target, voxel_size, performLocalRefinement):
# For documentation, see http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Fast-global-registration

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    
    if performLocalRefinement:
        start = time.time()
        result_FGR = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        print("FGR")
        print(result_FGR.transformation)
        result = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_FGR.transformation)
        total_time = time.time() - start
    else:
        start = time.time()
        result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        total_time = time.time() - start

    return result, total_time

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, global_transformation):
    # See documentation, http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Local-refinement
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, global_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

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

def readRegistrationProblem(datatext):
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
        registration_problem = [] # Create empty list for storing registration problems
        for i in range(1, len(lines)): # do not need header information, start at index 1
            line = lines[i].split()
            registration_problem.append(line) # reads the entire file
    for row in range(0, len(registration_problem)):
        for col in range(3,16):
            registration_problem[row][col] = float(registration_problem[row][col])

    return registration_problem

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
    print("\nRot Mat difference between experimental and true")
    print(diff_rot_mat)
    print("\nTranslation vector difference between experimental and true")
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

def main(algorithm, algorithm_type):

    # # Assume to be in parent folder of "point_clouds_registration_benchmark", these are folders to exclude
    # temp = ['devel', '__pycache__', '.git', 'metric', '.github', '.vscode', 'Algorithms', 'RESULTS']
    # # temp = temp + ['tum_(copy)', 'TEST'] # for debugging
    # temp = temp + ['eth', 'kaist', 'planetary', 'tum', 'tum_(copy)', 'Backup'] # Analyzing 'TEST' dataset only 
    # # temp = temp + ['eth', 'kaist', 'planetary', 'tum', 'tum_(copy)', 'TEST'] # for debugging
    # datasets_folders = sorted([folder_name for folder_name in os.listdir(".") if os.path.isdir(folder_name)])
    # datasets_folders = list(filter(lambda x : x not in temp, datasets_folders))

    # Specify which folders contain datasets to be analyzed
    # datasets_folders = ["eth", "kaist", "planetary", "tum"] # default dataset downloaded from github
    datasets_folders = ['TEST']

    # Create folder for storing results for each dataset
    if not os.path.exists('./' + 'RESULTS/' + algorithm + '_RESULTS/'):
        os.mkdir('./' + 'RESULTS/')
        os.mkdir('./' + 'RESULTS/' + algorithm + '_RESULTS/')
    else:
        Warning("\"RESULTS\" folder already exists. Previous results may have already been created and will be overwritten.")

    # Loop over dataset folders - ['eth', 'kaist', 'planetary', 'tum']
    for i in range(0, len(datasets_folders)):

        # Initialize results for each dataset folder
        temp_dir = os.path.join("./" + 'RESULTS/' + algorithm + "_RESULTS/", datasets_folders[i] + '_results')
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        # Change current directory to each dataset folder
        os.chdir(datasets_folders[i])
        print("Changing dataset directory to " + '\'' + datasets_folders[i] + '\'')
        # In each dataset folder, find all folders containing PCD files e.g. 'apartment' in 'eth' dataset folder
        datasets_i = sorted([folder_name for folder_name in os.listdir(".") if os.path.isdir(folder_name)]) 
        # In each dataset folder, find all folders containing each *.txt file containing info about point clouds and transformations
        if algorithm_type == "GLOBAL":
            datatext_i = sorted([file for file in os.listdir(".") if file.endswith('_global.txt')]) # for GLOBAL registration algorithms
        elif algorithm_type == "LOCAL":
            datatext_i = sorted([file for file in os.listdir(".") if file.endswith('_local.txt')]) # for LOCAL registration algorithms
        
        # Iterate through datatext list to read registration problems from _global.txt files
        for j in range(0, len(datasets_i)):

            # Initialize results for each dataset folder
            temp_dir = '../' + 'RESULTS/' + algorithm + '_RESULTS/' + datasets_folders[i] + '_results/'
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
                temp_dir = os.path.join(temp_dir + '/' + datatext_i[j].replace('.txt', '') + '_results.txt')
                open(temp_dir, 'w').close()
            else:
                temp_dir = os.path.join('../' + 'RESULTS/' + algorithm + '_RESULTS/' + datasets_folders[i] + '_results/'
                                            + datatext_i[j].replace('.txt', '') + '_results.txt')
                if os.path.exists(temp_dir):
                    open(temp_dir, 'w').close()

            results_file = open('../' + 'RESULTS/' + algorithm + '_RESULTS/' + datasets_folders[i] + '_results/'
                                    + datatext_i[j].replace('.txt', '') + '_results.txt', 'w')
            results_file.write('id_str src_str tgt_str overlap time ang_err err_met trans_err_vec\n')

            # Get the registration problem
            registration_problem = readRegistrationProblem(datatext_i[j]) # N by 16 multidimensional list

            os.chdir(datasets_i[j]) # Change directory to individual datasets, e.g 'apartment' in 'eth' dataset folder
            print("------> Changing to individual scenario: " + '\'' + datasets_i[j] + '\'')
            for row in registration_problem:
                # Get required information to set up each registration problem
                id_str = row[0] # Unique identifier of the registration problem
                src_str = row[1] # File name of the Source Point Cloud
                tgt_str = row[2] # File name of the Target Point Cloud
                overlap = row[3] # Percentage of overlap between the two point clouds
                true_transformation = np.array([[row[4],  row[5],  row[6],  row[7]], # Affine 4x4 transformation matrix
                                        [row[8],  row[9],  row[10], row[11]],
                                        [row[12], row[13], row[14], row[15]],
                                        [0,       0,       0,       1]]) 
                true_rot_mat, true_translation = getRotMatAndTransVec(true_transformation)
                print("True transformation is:")
                print(true_transformation)
                ## Now read in raw Point Cloud Data *.pcd files as specified for each registration problem ##

                # Get the source point cloud to be transformed
                src_pcd = o3d.io.read_point_cloud(src_str)
                src_pcd_transformed = copy.deepcopy(src_pcd) # perform copy so that original source point cloud is preserved
                src_pcd_transformed = src_pcd_transformed.transform(true_transformation) # transform source point cloud with initial perturbation as defined in registration problem
                # Load in source point cloud as an array
                src_pc_transformed = pcd2xyz(src_pcd_transformed)

                # Get the target point cloud
                tgt_pcd = o3d.io.read_point_cloud(tgt_str) # Load in target point cloud
                # Load in target point cloud as an array
                tgt_pc = pcd2xyz(tgt_pcd)

                # Now align the point clouds with the specific algorithm to test
                if algorithm_type == "LOCAL":
                    if algorithm == "ICP_point2point":
                        ##########################
                        # ICP Point-to-Point Algorithm Call
                        ##########################
                        # Setup input threshold
                        threshold = 0.02
                        # Get Registration Solution
                        solution, alg_time = ICP_point2point(src_pcd, tgt_pcd, threshold, true_transformation)
                        T_target2source = solution.transformation
                        T_source2target = np.linalg.inv(solution.transformation)
                        print("ICP Point-to-Point took %0.5f seconds" %alg_time)
                        print("Registration solution is:")
                        print(T_target2source)
                        printError(true_rot_mat, true_translation, T_target2source)
                    elif algorithm == "ICP_point2plane":
                        ##########################
                        # ICP Point-to-Plane Algorithm Call
                        ##########################
                        # Setup input threshold
                        threshold = 0.02
                        # Get Registration Solution
                        solution, alg_time = ICP_point2plane(src_pcd, tgt_pcd, threshold, true_transformation)
                        T_target2source = solution.transformation
                        T_source2target = np.linalg.inv(solution.transformation)
                        print("ICP Point-to-Plane took %0.5f seconds" %alg_time)
                        print("Registration solution is:")
                        print(T_target2source)
                        printError(true_rot_mat, true_translation, T_target2source)
                elif algorithm_type == "GLOBAL":
                    if algorithm == "FGR":
                        ###########################
                        ## Fast Global Registration Call
                        ###########################
                        # This implementation of FGR returns the affine transformation matrix to transform 
                        # from the initially perturbed source point cloud to the target point cloud
                        # Solution from source to target

                        # Get input voxel size for desired downsampling
                        voxel_size = 0.1

                        # Perform a local registration algorithm i.e. ICP Point-to-Point after rough 
                        # alignment from global registration algorithm
                        performLocalRefinement = False # not working at the moment

                        # Get Registration Solution 
                        solution, alg_time = execute_fast_global_registration(src_pcd_transformed, tgt_pcd, voxel_size, performLocalRefinement)
                        T_source2target = solution.transformation
                        T_target2source = np.linalg.inv(solution.transformation)
                        print("FGR took %0.2f seconds" %alg_time)
                        print("Registration Solution from target to source is:")
                        print(T_target2source)
                        printError(true_rot_mat, true_translation, T_target2source)   
                    elif algorithm == "Go-ICP":
                        ###########################
                        ## Go-ICP Algorithm Call
                        ###########################
                        
                        # ## Voxel Downsampling
                        # To increase registration speed, we perform voxel downsampling and visualize the downsampled point clouds.
                        voxel_size = 0.1
                        src_pcd_transformed_down, tgt_pcd_down = voxel_downsampling(src_pcd_transformed, tgt_pcd, voxel_size)

                        # Go-ICP returns transformation solution from source to target
                        solution, alg_time = goICP(pcd2xyz(tgt_pcd_down), pcd2xyz(src_pcd_transformed_down))
                        T_source2target = solution
                        T_target2source = np.linalg.inv(solution) 
                        print("Go-ICP took %0.2f seconds" %(alg_time))
                        print("Go-ICP solution from source to target is:")
                        print(solution)
                        printError(true_rot_mat, true_translation, solution)
                    elif algorithm == "RANSAC":
                        ###########################
                        ## RANSAC Global Registration Call
                        ###########################
                        # This implementation of RANSAC returns the affine transformation matrix to transform 
                        # from the initially perturbed source point cloud back to the target point cloud
                        # Solution from source to target

                        # Get input voxel size for desired downsampling
                        voxel_size = 0.1 # means 10 cm for the dataset

                        # Perform a local registration algorithm i.e. ICP Point-to-Point after rough 
                        # alignment from global registration algorithm
                        performLocalRefinement = False # not working at the moment

                        # Get Registration Solution
                        solution, alg_time = execute_fast_global_registration(src_pcd_transformed, tgt_pcd, voxel_size, performLocalRefinement)
                        T_source2target = solution.transformation
                        T_target2source = np.linalg.inv(solution.transformation)

                        print("RANSAC took %0.2f seconds" %alg_time)
                        print("Registration Solution from target to source is:")
                        print(T_target2source)
                        printError(true_rot_mat, true_translation, T_target2source)
                    elif algorithm == "TEASER":
                        pass

                # ## Get RESULTS ###
                # # Calculate Error
                # _, trans_err = getError(true_rot_mat, true_translation, T_target2source)
                # rot_exp, _ = getRotMatAndTransVec(T_target2source)
                # ang_err = get_angular_error(rot_exp, true_rot_mat)
                
                # # Compare the aligned source point cloud with the original source point cloud using their error metric
                # # This reduces the ambiguity with an individual rotation and translation error
                # # if ~0, then no error
                # src_pcd_restored = copy.deepcopy(src_pcd_transformed)
                # err_met = calculate_error(src_pcd_restored.transform(T_source2target), src_pcd)

                # # Write to output results files to be saved and analyzed within MATLAB/Python             
                # results_file.write(id_str + " " + src_str + " " + tgt_str + " " + str(overlap) + " "
                #                      + str(alg_time) + " " + str(ang_err) + " " + str(err_met) + " " 
                #                      + str(trans_err[0]) + " " + str(trans_err[1]) + " " 
                #                      + str(trans_err[2]) + "\n")

                # # View registration problem (source in blue, target in red)
                # draw_registration_result(src_pcd, tgt_pcd, true_transformation)
                # # View registration solution (source in blue, target in red)
                # draw_registration_result(src_pcd_transformed, tgt_pcd, T_source2target)
                # # draw_registration_result(tgt_pcd, src_pcd_transformed, T_target2source)

            results_file.close()
            os.chdir("..") # Change directory back to parent dataset folder, e.g. 'eth'
    
        # Go back to parent directory "point_cloud_registration_benchmark" folder and get ready to initialize next dataset folder
        os.chdir('..')

if __name__ == '__main__':
    
    # Time the  project
    start_time = time.time()

    # Setup initial parameters
    local_registration_algs = ["ICP_point2point", "ICP_point2plane"] # local registration algorithms
    global_registration_algs = ["FGR", "Go-ICP", "TEASER", "RANSAC"] # global registration algorithms

    # Now select whether local or global registration problems are to be tested
    # If "GLOBAL", contains registration problems that include a single perturbation to the source point cloud, use global registration algorihtms
    # If "LOCAL", contains registration problems that include multiple perurbations to the source point cloud, use local registration algorithms
    algorithm_type = "LOCAL" 

    # For simplicity purposes, only test a single algorithm at a time
    algorithm = local_registration_algs[0]
    # algorithm = global_registration_algs[2]

    # Call main function
    main(algorithm, algorithm_type)

    # TEASER_example()

    end_time = time.time()
    total_time = end_time - start_time
    print('DONE')
    print('Total time completed in ' + str(total_time) + ' seconds')

