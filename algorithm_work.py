                



                # ###########################
                # ## Fast Global Registration Call 2
                # ###########################
                # # This implementation of FGR returns the affine transformation matrix to transform 
                # # from the initially perturbed source point cloud to the target point cloud
                # # Solution from source to target

                # # Get input voxel size for desired downsampling
                # voxel_size = 0.05

                # # Get Registration Solution
                # alg_start = time.time()
                # solution = execute_fast_global_registration(src_pcd_transformed, tgt_pcd, voxel_size)

                # complete_solution = np.copy(solution.transformation)
                # src_pcd_transformed_new = copy.deepcopy(src_pcd_transformed)
                # for ndx in range(4):
                #     # draw_registration_result(src_pcd_transformed_new, tgt_pcd, solution.transformation)
                #     src_pcd_transformed_new = src_pcd_transformed_new.transform(solution.transformation)
                #     solution = execute_fast_global_registration(src_pcd_transformed_new, tgt_pcd, voxel_size)
                    
                #     solRot, solTrans = getRotMatAndTransVec(solution.transformation)
                #     compRot, compTrans = getRotMatAndTransVec(complete_solution)

                #     complete_Rot = np.matmul(solRot, compRot)
                #     complete_Trans = solTrans + compTrans

                #     complete_solution = convert2affineTrans(complete_Rot, complete_Trans)

                # alg_stop = time.time()
                # print("FGR took %0.2f seconds" %(alg_stop - alg_start))
                # print("Registration Solution is:")
                # print(complete_solution)
                # # solRot, solTrans = getRotMatAndTransVec(complete_solution)
                # # sol_transformation = convert2affineTrans(np.transpose(solRot), -solTrans)
                # sol_transformation = transposeAffine(complete_solution)
                # printError(true_rot_mat, true_translation, sol_transformation)
                # draw_registration_result(src_pcd_transformed, tgt_pcd, np.eye(4,4))
                # draw_registration_result(src_pcd_transformed, tgt_pcd, complete_solution)

                # ###########################
                # ## Fast Global Registration Call + Point-to-Plane ICP
                # ###########################

                # # Use FGR on a heavily down-sampled point cloud and then use ICP algorithm to 
                # # further refine the alignment

                # # View registration problem (source in blue, target in red)
                # draw_registration_result(src_pcd, tgt_pcd, true_transformation)

                # ICP_point2point_opt = True 
                # ICP_point2plane_opt = False 

                # # Get input voxel size for desired downsampling
                # start_time = time.time()
                # voxel_size = 0.1

                # # Get FGR Registration Solution on heavily downsampled point cloud
                # FGR_solution = execute_fast_global_registration(src_pcd_transformed, tgt_pcd, voxel_size)
                # FGR_stop = time.time()
                # print("FGR took %0.2f seconds" %(FGR_stop - start_time))
                # draw_registration_result(src_pcd_transformed, tgt_pcd, FGR_solution.transformation)
                # print("FGR Solution is: ")
                # print(FGR_solution.transformation)

                # src_pcd_transformed_FGR = copy.deepcopy(src_pcd_transformed.transform(FGR_solution.transformation))

                # if ICP_point2point_opt:
                #     ##  ICP Point-to-Point Algorithm Call ##
                #     # Setup input threshold
                #     threshold = 0.02

                #     # Get Registration Solution
                #     ICP_start = time.time()
                #     ICP_solution = ICP_point2point(src_pcd_transformed, tgt_pcd, threshold, FGR_solution.transformation)
                #     # ICP_solution = ICP_point2point(src_pcd_transformed_FGR, tgt_pcd, threshold,
                #     #                 np.matmul(true_transformation, np.linalg.inv(FGR_solution.transformation)))

                #     ICP_stop = time.time()
                #     print("ICP Point-to-Point took %0.5f seconds" %(ICP_stop - ICP_start))
                #     print("ICP Registration Solution is:")
                #     print(ICP_solution.transformation)
                #     draw_registration_result(src_pcd_transformed_FGR, tgt_pcd, ICP_solution.transformation)
                #     # draw_registration_result(src_pcd_transformed_FGR, tgt_pcd, np.linalg.inv(ICP_solution.transformation))

                #     complete_solution = np.matmul(ICP_solution.transformation, FGR_solution.transformation)
                #     print('Complete solution is:')
                #     print(complete_solution)
                #     printError(true_rot_mat, true_translation, complete_solution)

                #     # View registration solution (source in blue, target in red)
                #     # draw_registration_result(src_pcd.transform(true_transformation), tgt_pcd, np.linalg.inv(complete_solution))

                #     stop_time = time.time()
                #     print("FGR + ICP Point to Point took %0.2f seconds" %(stop_time - start_time))

                # elif ICP_point2plane_opt:
                #     print("")

# Miscellaneous Work
# # Translation Only
# a = pcd2xyz(copy.deepcopy(src_pcd))
# a = a + translation
# a = xyz2pcd(a)
# b = src_pcd.transform(convert2affineTrans(np.array([[1,0,0],[0,1,0],[0,0,1]]), translation))
# a.paint_uniform_color([0, 0, 1]) # blue
# b.paint_uniform_color([1, 0, 0]) # red
# o3d.visualization.draw_geometries([a, b])
# print(np.argwhere(pcd2xyz(a) != pcd2xyz(b)))

# # Rotation Only
# a = pcd2xyz(src_pcd)
# a = np.matmul(a, rot_mat)
# a = xyz2pcd(a)
# new_transformation = convert2affineTrans(np.transpose(rot_mat), np.array([0,0,0]))
# b = src_pcd.transform(new_transformation) # transform 
# a.paint_uniform_color([0, 0, 1]) # blue
# b.paint_uniform_color([1, 0, 0]) # red
# o3d.visualization.draw_geometries([a, b])
# # print(np.argwhere(pcd2xyz(a) != pcd2xyz(b)))
# # print(np.argwhere((pcd2xyz(a) - pcd2xyz(b)) > 1E-15))
# print(np.max(pcd2xyz(a) - pcd2xyz(b)))

# # Full Affine Transformation
# a = pcd2xyz(copy.deepcopy(src_pcd))
# a = np.matmul(a, np.transpose(rot_mat)) + translation
# b = pcd2xyz(src_pcd.transform(transformation))
# pcd_a = xyz2pcd(a)
# pcd_b = xyz2pcd(b)
# pcd_a.paint_uniform_color([0, 0, 1]) # blue
# pcd_b.paint_uniform_color([1, 0, 0]) # red
# o3d.visualization.draw_geometries([pcd_a, pcd_b])

# Miscellaneous
                # voxel_size = 0.5
                # source_down, source_fpfh = preprocess_point_cloud(src_pcd_transformed, voxel_size)
                # target_down, target_fpfh = preprocess_point_cloud(tgt_pcd, voxel_size)
                # alg_start = time.time()
                # FGR_solution = execute_fast_global_registration(source_down, target_down,
                #                                source_fpfh, target_fpfh,
                #                                voxel_size)
                # alg_stop = time.time()
                # print("FGR took %0.2f seconds" %(alg_stop - alg_start))
                # threshold = 0.02
                # solution = ICP_point2plane(src_pcd, tgt_pcd, threshold, FGR_solution.transformation)
                # alg_stop = time.time()
                # print("Point-to-Plane ICP took %0.2f seconds" %(alg_stop - alg_start))
                # print("Registration Solution is:")
                # print(solution.transformation)
                # printError(true_rot_mat, true_translation, solution.transformation)

                # target_temp = copy.deepcopy(tgt_pcd)
                # source_temp = copy.deepcopy(src_pcd_transformed)
                # target_temp.transform(solution.transformation)
                # target_temp.paint_uniform_color([1, 0, 0]) # red
                # source_temp.paint_uniform_color([0, 0, 1]) # blue

                # o3d.visualization.draw_geometries([src_pcd_transformed, target_temp])
