

                # ##########################
                # # ICP Point-to-Plane Algorithm Call
                # ##########################
                # # Setup input threshold
                # threshold = 0.02
                # # Get Registration Solution
                # solution, alg_time = ICP_point2plane(src_pcd, tgt_pcd, threshold, true_transformation)
                # T_target2source = solution.transformation
                # T_source2target = np.linalg.inv(solution.transformation)
                # print("ICP Point-to-Plane took %0.5f seconds" %alg_time)
                # print("Registration solution from target to source is:")
                # print(T_target2source)
                # printError(true_rot_mat, true_translation, T_target2source)

                # ###########################
                # ## Fast Global Registration Call
                # ###########################
                # # This implementation of FGR returns the affine transformation matrix to transform 
                # # from the initially perturbed source point cloud to the target point cloud
                # # Solution from source to target

                # # Get input voxel size for desired downsampling
                # voxel_size = 0.1

                # # Perform a local registration algorithm i.e. ICP Point-to-Point after rough 
                # # alignment from global registration algorithm
                # performLocalRefinement = False # not working at the moment

                # # Get Registration Solution 
                # solution, alg_time = execute_fast_global_registration(src_pcd_transformed, tgt_pcd, voxel_size, performLocalRefinement)
                # T_source2target = solution.transformation
                # T_target2source = np.linalg.inv(solution.transformation)
                # print("FGR took %0.2f seconds" %alg_time)
                # print("Registration Solution from target to source is:")
                # print(T_target2source)
                # printError(true_rot_mat, true_translation, T_target2source)   

                # ###########################
                # ## Go-ICP Algorithm Call
                # ###########################
                
                # # ## Voxel Downsampling
                # # To increase registration speed, we perform voxel downsampling and visualize the downsampled point clouds.
                # voxel_size = 0.1
                # src_pcd_transformed_down, tgt_pcd_down = voxel_downsampling(src_pcd_transformed, tgt_pcd, voxel_size)

                # # Go-ICP returns transformation solution from source to target
                # solution, alg_time = goICP(pcd2xyz(tgt_pcd_down), pcd2xyz(src_pcd_transformed_down))
                # T_source2target = solution
                # T_target2source = np.linalg.inv(solution) 
                # print("Go-ICP took %0.2f seconds" %(alg_time))
                # print("Go-ICP solution from source to target is:")
                # print(solution)
                # printError(true_rot_mat, true_translation, solution)

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