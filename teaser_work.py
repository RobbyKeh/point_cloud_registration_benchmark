def TEASER(src, dst):
# TEASER++ is a correspondence-based algorithm and it takes two numpy arrays of 
# equal number of columns, 3 x N, where N is the number of matches (not number of 
# points in the original point clouds). Column i of the first array (a 3D point) 
# corresponds to column i of the second array (another 3D point).

    # Populate the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    # solver_params.cbar2 = 1
    # solver_params.noise_bound = 0.01
    # solver_params.estimate_scaling = True
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    # solver_params.rotation_gnc_factor = 1 #1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    print("TEASER++ Parameters are:", solver_params)
    
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    teaserpp_solver.solve(src, dst) # source, transformed data

    solution = teaserpp_solver.getSolution()

    # Print the solution
    print("Solution is:", solution)

def TEASER_test():

    # Generate random data points
    src = np.random.rand(3, 20000)

    # Apply arbitrary scale, translation and rotation
    scale = 1.5
    translation = np.array([[1], [0], [-1]])
    rotation = np.array([[0.98370992, 0.17903344,   -0.01618098],
                        [-0.04165862, 0.13947877,   -0.98934839],
                        [-0.17486954, 0.9739059,    0.14466493]])
    dst = scale * np.matmul(rotation, src) + translation

    # Add two outliers
    dst[:, 1] += 10
    dst[:, 9] += 15

    # Populate the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.01
    solver_params.estimate_scaling = True
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver   .ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    print("TEASER++ Parameters are:", solver_params)
    
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    teaserpp_solver.solve(src, dst)

    solution = teaserpp_solver.getSolution()

    # Print the solution
    print("Solution is:", solution)

def TEASER_example():
    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    os.chdir("/home/robby/Desktop/TEASER-plusplus/examples/")
    NOISE_BOUND = 0.05
    N_OUTLIERS = 1700
    OUTLIER_TRANSLATION_LB = 5
    OUTLIER_TRANSLATION_UB = 10

    # Load bunny ply file
    src_cloud = o3d.io.read_point_cloud("./example_data/bun_zipper_res3.ply")
    o3d.visualization.draw_geometries([src_cloud])

    src = np.transpose(np.asarray(src_cloud.points))
    N = src.shape[1]

    # Apply arbitrary scale, translation and rotation
    T = np.array(
        [[9.96926560e-01, 6.68735757e-02, -4.06664421e-02, -1.15576939e-01],
         [-6.61289946e-02, 9.97617877e-01, 1.94008687e-02, -3.87705398e-02],
         [4.18675510e-02, -1.66517807e-02, 9.98977765e-01, 1.14874890e-01],
         [0, 0, 0, 1]])

    dst_cloud = copy.deepcopy(src_cloud)
    dst_cloud.transform(T)
    dst = np.transpose(np.asarray(dst_cloud.points))

    # Add some noise
    dst += (np.random.rand(3, N) - 0.5) * 2 * NOISE_BOUND

    # Add some outliers
    outlier_indices = np.random.randint(N_OUTLIERS, size=N_OUTLIERS)
    for i in range(outlier_indices.size):
        shift = OUTLIER_TRANSLATION_LB + np.random.rand(3, 1) * (OUTLIER_TRANSLATION_UB - OUTLIER_TRANSLATION_LB)
        dst[:, outlier_indices[i]] += shift.squeeze()
    
    # o3d.visualization.draw_geometries([dst_cloud])
    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(np.transpose(dst))
    o3d.visualization.draw_geometries([dst_pcd])

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    solver.solve(src, dst)
    end = time.time()

    solution = solver.getSolution()

    true_rot_mat, true_translation = getRotMatAndTransVec(T)
    diff_rot_mat = np.matmul(true_rot_mat, np.transpose(solution.rotation))
    diff_trans_vec = true_translation - solution.translation

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")

    print("Expected rotation: ")
    print(T[:3, :3])
    print("Estimated rotation: ")
    print(solution.rotation)
    print("Rot Mat Difference")
    print(diff_rot_mat)
    print("Error (rad): ")
    print(get_angular_error(T[:3,:3], solution.rotation))

    print("Expected translation: ")
    print(T[:3, 3])
    print("Estimated translation: ")
    print(solution.translation)
    print("Translation Vector Difference")
    print(diff_trans_vec)
    print("Error (m): ")
    print(np.linalg.norm(T[:3, 3] - solution.translation))

    print("Number of correspondences: ", N)
    print("Number of outliers: ", N_OUTLIERS)
    print("Time taken (s): ", end - start)

    # View transformed point cloud with target point cloud
    src_cloud.paint_uniform_color([1, 0, 0]) # show source in red
    dst_cloud.paint_uniform_color([0, 0, 1]) # show target in blue
    # o3d.visualization.draw_geometries([src_cloud, dst_cloud])
    draw_registration_result(dst_cloud, src_cloud, np.eye(4,4)) 
    # show source in blue
    # show target in red
    
    # View registration solution
    # draw_registration_result(dst_cloud, src_cloud, convert2affineTrans(solution.rotation, solution.translation))
    # show source in blue
    # show target in red

def TEASER2(src_cloud, dst_cloud):
    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    NOISE_BOUND = 0.05
    N_OUTLIERS = 1700
    OUTLIER_TRANSLATION_LB = 5
    OUTLIER_TRANSLATION_UB = 10

    # o3d.visualization.draw_geometries([src_cloud])

    src = np.transpose(np.asarray(src_cloud.points))
    N = src.shape[1]
    dst = np.transpose(np.asarray(dst_cloud.points))

    # # Add some noise
    # dst += (np.random.rand(3, N) - 0.5) * 2 * NOISE_BOUND

    # # Add some outliers
    # outlier_indices = np.random.randint(N_OUTLIERS, size=N_OUTLIERS)
    # for i in range(outlier_indices.size):
    #     shift = OUTLIER_TRANSLATION_LB + np.random.rand(3, 1) * (OUTLIER_TRANSLATION_UB - OUTLIER_TRANSLATION_LB)
    #     dst[:, outlier_indices[i]] += shift.squeeze()
    
    # o3d.visualization.draw_geometries([dst_cloud])
    # dst_pcd = o3d.geometry.PointCloud()
    # dst_pcd.points = o3d.utility.Vector3dVector(np.transpose(dst))
    # o3d.visualization.draw_geometries([dst_pcd])

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    solver.solve(src, dst)
    end = time.time()

    solution = solver.getSolution()

    # print("Number of correspondences: ", N)
    # print("Number of outliers: ", N_OUTLIERS)
    print("Time taken (s): ", end - start)

    return solution

def TEASER3(src_cloud, dst_cloud):
    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    # NOISE_BOUND = 0.05
    # N_OUTLIERS = 1700
    # OUTLIER_TRANSLATION_LB = 5
    # OUTLIER_TRANSLATION_UB = 10

    src = np.transpose(np.asarray(src_cloud.points))
    N = src.shape[1]
    dst = np.transpose(np.asarray(dst_cloud.points))

    # # Add some noise
    # dst += (np.random.rand(3, N) - 0.5) * 2 * NOISE_BOUND

    # # Add some outliers
    # outlier_indices = np.random.randint(N_OUTLIERS, size=N_OUTLIERS)
    # for i in range(outlier_indices.size):
    #     shift = OUTLIER_TRANSLATION_LB + np.random.rand(3, 1) * (OUTLIER_TRANSLATION_UB - OUTLIER_TRANSLATION_LB)
    #     dst[:, outlier_indices[i]] += shift.squeeze()
    
    # o3d.visualization.draw_geometries([dst_cloud])
    # dst_pcd = o3d.geometry.PointCloud()
    # dst_pcd.points = o3d.utility.Vector3dVector(np.transpose(dst))
    # o3d.visualization.draw_geometries([dst_pcd])

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    # solver_params.cbar2 = 1
    # solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    # solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    solver.solve(src, dst)
    end = time.time()

    solution = solver.getSolution()

    # print("Number of correspondences: ", N)
    # print("Number of outliers: ", N_OUTLIERS)
    print("Time taken (s): ", end - start)

    return solution

def execute_teaser_global_registration(source, target):
    """
    Use TEASER++ to perform global registration
    """

    NOISE_BOUND = 0.05

    # Prepare TEASER++ Solver
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    print("TEASER++ Parameters are:", solver_params)
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    # Solve with TEASER++
    start = time.time()
    teaserpp_solver.solve(source, target)
    end = time.time()
    est_solution = teaserpp_solver.getSolution()
    # est_mat = bench_utils.compose_mat4_from_teaserpp_solution(est_solution)
    # max_clique = teaserpp_solver.getTranslationInliersMap()
    # print("Max clique size:", len(max_clique))
    # final_inliers = teaserpp_solver.getTranslationInliers()
    # return est_mat, max_clique, end - start

    return est_solution

                # ##########################
                # # TEASER Algorithm Call (Global), use this one
                # ##########################
                # # # Note: TEASER algorithm requires input point clouds to be of the same length
                
                # sameSize = False
                # if sameSize:
                #     src_pc_transformed_TEASER, tgt_pc_TEASER = convertArraysToSameDimension(copy.deepcopy(src_pc_transformed), copy.deepcopy(tgt_pc))
                # else:
                #     tgt_pc_TEASER = copy.deepcopy(tgt_pc)
                #     src_pc_transformed_TEASER = copy.deepcopy(src_pc_transformed)

                # # ## Voxel Downsampling
                # # To increase registration speed, we perform voxel downsampling and visualize the downsampled point clouds.
                # voxel_size = 0.1
                # src_pcd_transformed_TEASER, tgt_pcd_TEASER = voxel_downsampling(src_pcd_transformed, tgt_pcd, voxel_size)

                # # Get the solution
                # # solution = TEASER3(src_pcd_transformed_TEASER, tgt_pcd_TEASER)
                # solution = execute_teaser_global_registration(src_pcd_transformed_TEASER, tgt_pcd_TEASER)
                # est_transformation = convert2affineTrans(solution.rotation, solution.translation)
                # # est_transformation = np.linalg.inv(est_transformation)
                # print("=====================================")
                # print("          TEASER++ Results           ")
                # print("=====================================")
                # # print("Estimated rotation: ")
                # # print(solution.rotation)
                # # print("Estimated translation: ")
                # # print(solution.translation)
                # print("Estimated Transformation: ")
                # print(est_transformation)
                # printError(true_rot_mat, true_translation, est_transformation)

                # # View transformed point cloud with target point cloud
                # draw_registration_result(src_pcd, tgt_pcd, true_transformation) # registration problem

                # # View registration solution
                # draw_registration_result(src_pcd_transformed_TEASER, tgt_pcd_TEASER, est_transformation)
                # # show source in blue
                # # show target in red



                # ###########################
                # ## TEASER Algorithm Call 
                # ###########################
                # # Note: TEASER algorithm requires input point clouds to be of the same length
                
                # # Get the source point cloud to be transformed
                # src_pcd = o3d.io.read_point_cloud(src_str)
                # src_pcd_transformed = copy.deepcopy(src_pcd)
                # src_pcd_transformed = src_pcd_transformed.transform(true_transformation)
                # # Load in source point cloud as an array
                # src_pc_transformed = pcd2xyz(src_pcd_transformed)
                # # Get the target point cloud
                # tgt_pcd = o3d.io.read_point_cloud(tgt_str) # Load in target point cloud
                # tgt_pc = pcd2xyz(tgt_pcd)

                # src_pc_transformed_TEASER = copy.deepcopy(src_pc_transformed)
                # tgt_pc_TEASER = copy.deepcopy(tgt_pc)
                
                # src_len = len(src_pc_transformed_TEASER)
                # tgt_len = len(tgt_pc_TEASER)
                
                # beg = time.time()
                # print("Size of Source Point Cloud BEFORE random point deletion: %d" %src_len)
                # print("Size of Target Point Cloud BEFORE random point deletion: %d" %tgt_len)
                # print("Deleting random points to get equal dimensions of source and target point clouds....")
                # while src_len != tgt_len:
                #     src_len = len(src_pc_transformed_TEASER)
                #     tgt_len = len(tgt_pc_TEASER)
                #     if src_len > tgt_len:
                #         rand_ndx = np.random.randint(0, src_len)
                #         src_pc_transformed_TEASER = np.delete(src_pc_transformed_TEASER, rand_ndx, axis=0) # Delete random row
                #     else:
                #         rand_ndx = np.random.randint(0, tgt_len)
                #         tgt_pc_TEASER = np.delete(tgt_pc_TEASER, rand_ndx, axis=0) # Delete random row
                # print("This took %0.4f seconds" %(time.time() - beg))
                # print("Size of Source Point Cloud AFTER random point deletion: %d" %len(src_pc_transformed_TEASER))
                # print("Size of Target Point Cloud AFTER random point deletion: %d" %len(tgt_pc_TEASER))

                # ds_size = 10
                # src_pcd_transformed_TEASER = o3d.geometry.PointCloud.uniform_down_sample(xyz2pcd(src_pc_transformed_TEASER), ds_size)
                # tgt_pcd_TEASER = o3d.geometry.PointCloud.uniform_down_sample(xyz2pcd(tgt_pc_TEASER), ds_size)

                # src_pc_transformed_TEASER = pcd2xyz(src_pcd_transformed_TEASER)
                # tgt_pc_TEASER = pcd2xyz(tgt_pcd_TEASER)
                # print("Size of Source Point Cloud AFTER downsampling: %d" %len(src_pc_transformed_TEASER))
                # print("Size of Target Point Cloud AFTER downsampling: %d" %len(tgt_pc_TEASER))
                # TEASER(np.transpose(src_pc_transformed_TEASER), np.transpose(tgt_pc_TEASER))