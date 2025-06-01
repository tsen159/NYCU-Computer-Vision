import cv2
import numpy as np

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
def create_im_window(window_name, img):
    cv2.imshow(window_name, img)

# show the all window you call before im_show()
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows() # press any key to close all windows

def get_kp_des(img_gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    return kp, des

def knn_point_matcher(kp1, kp2, des1_list, des2_list, k=2, threshold=0.75):
    """
    Get the matched points between two images.

    Args:
        des_list_1: the descriptors of image 1
        des_list_2: the descriptors of image 2
        k: the number of nearest neighbors
        threshold: the threshold of Lowe's ratio test
    Returns:
        matched_points: list, the indices of matched points between two images
    """
    matched_points = []

    for i in range(len(des1_list)):
        des1 = des1_list[i]
        dist_list = np.linalg.norm(des2_list - des1, axis=1) # compute the distance between the descriptors
        match_list = [[j, dist_list[j]] for j in range(len(dist_list))] # the index and distance of each point
        match_list.sort(key=lambda x: x[1])
        match_list = match_list[:k]

        # Lowe's ratio test
        if match_list[0][1] < threshold * match_list[1][1]: 
            pos1 = [int(kp1[i].pt[0]), int(kp1[i].pt[1])] # position of the point in img_1
            pos2 = [int(kp2[match_list[0][0]].pt[0]), int(kp2[match_list[0][0]].pt[1])] # position of the point in img_2
            matched_points.append([pos1, pos2])

    #print(np.array(matched_points).shape)
    
    return matched_points


def get_homography(pts_src, pts_tar):
    """
    Solve the homography matrix H of two given images.

    Args:
        pts_src: list, the positions of matched points in source image 
        pts_des: list, the positions of matched points in target image 
    Returns:
        H: the homography matrix
    """
    A = np.zeros((2*len(pts_tar), 9))
    for i in range(len(pts_tar)):
        A[2*i] = [-pts_src[i][0], -pts_src[i][1], -1, 0, 0, 0, pts_tar[i][0]*pts_src[i][0], pts_tar[i][0]*pts_src[i][1], pts_tar[i][0]]
        A[2*i+1] = [0, 0, 0, -pts_src[i][0], -pts_src[i][1], -1, pts_tar[i][1]*pts_src[i][0], pts_tar[i][1]*pts_src[i][1], pts_tar[i][1]]

    U, S, V = np.linalg.svd(A) # SVD decomposition of A
    # the last row of V is the solution of H
    H = V[-1].reshape((3, 3))
    H = H / H[-1, -1] # normalization
    return H

def RANSAC(matched_points, threshold=5):
    """
    Use RANSAC to find the best homography matrix.
    Args:
        matched_points: list, the positions of matched points, shape (N, 2, 2)
        threshold: the threshold of distance between the predicted position and the real position
    Returns:
        best_H: the best homography matrix
    """
    num_iter = 0
    max_inlier_num = 0
    best_H = None
    max_inlier_list = []

    while True:
        num_iter += 1

        # randomly select 4 pairs of matched points
        idx = np.random.randint(len(matched_points), size=8)
        pts_src = [matched_points[i][0] for i in idx]
        pts_tar = [matched_points[i][1] for i in idx]
        H = get_homography(pts_src, pts_tar)

        inlier_num = 0
        inlier_list = []
        for (pos1, pos2) in matched_points:
            # transfrom to 3D
            pos_src = np.array([pos1[0], pos1[1], 1])
            pos_des = np.array([pos2[0], pos2[1], 1])
            pos_des_pred = np.dot(H, pos_src) # predict the position of the point in target image
            pos_des_pred = pos_des_pred / pos_des_pred[-1] # normalization to make z = 1
            if np.linalg.norm(pos_des - pos_des_pred) < threshold:
                inlier_num += 1
                inlier_list.append([pos1, pos2])

        # update the best homography matrix
        if inlier_num > max_inlier_num:
            max_inlier_num = inlier_num
            best_H = H
            max_inlier_list = inlier_list

        if num_iter >= 1000: break

    # recompute the homography matrix with all inliers
    pts_src = [max_inlier_list[i][0] for i in range(len(max_inlier_list))]
    pts_tar = [max_inlier_list[i][1] for i in range(len(max_inlier_list))]
    best_H = get_homography(pts_src, pts_tar)

    return best_H

def get_new_corners(img_src, H):
    """
    Get the transformed corner of the source image.
    Args:
        img_src: the source image
        H: the homography matrix
    Returns:
        new_corners: the coordinates of the 4 corners of the src image
    """
    src_h, src_w = img_src.shape[:2]
    corners = np.zeros((4, 3))
    corners[0] = H @ np.array([0, 0, 1])
    corners[1] = H @ np.array([src_w-1, 0, 1])
    corners[2] = H @ np.array([0, src_h-1, 1])
    corners[3] = H @ np.array([src_w-1, src_h-1, 1])
    src_corners = corners / corners[:,-1].reshape((4, 1)) # normalization

    return src_corners

def warp_images(img_src, img_tar, H):
    """
    Transform both source image and target image to the same coordinate system.

    Args:
        img_src: the source image
        img_tar: the target image
        H: the homography matrix
    Returns:
        img_src_warped: the source image after warping
        img_tar_trans: the target image after translation
    """
    new_corners = get_new_corners(img_src, H) # new corners of the source image
    new_origin_x = min(min(new_corners[0][0], new_corners[2][0]), 0)
    new_origin_y = min(min(new_corners[0][1], new_corners[1][1]), 0)

    # compute the size of the new image
    tar_h, tar_w = img_tar.shape[:2]
    new_w = max(max(new_corners[1][0], new_corners[3][0]), tar_w-1) - new_origin_x + 1
    new_h = max(max(new_corners[2][1], new_corners[3][1]), tar_h-1) - new_origin_y + 1
    new_w = int(new_w)
    new_h = int(new_h)

    T = np.array([[1, 0, np.abs(new_origin_x)],
                  [0, 1, np.abs(new_origin_y)], 
                  [0, 0, 1]]).astype(float) # translation matrix
    
    img_tar_trans = cv2.warpPerspective(img_tar, T, (new_w, new_h))
    img_src_warped = cv2.warpPerspective(img_src, T @ H, (new_w, new_h))

    return img_src_warped, img_tar_trans


class Blender():
    def direct_blending(self, img_src, img_tar):
        """
        Directly blend two images by averaging the overlapping area.
        """
        h, w = img_tar.shape[0], img_tar.shape[1]
        result = img_tar.copy()

        for y in range(h):
            for x in range(w):
                if (img_tar[y, x] != 0).any() and (img_src[y, x] != 0).any():
                    result[y, x] = (img_src[y, x] / 2 + img_tar[y, x] / 2).astype(np.uint8)
                elif (img_tar[y, x] != 0).any():
                    result[y, x] = img_tar[y, x]
                elif (img_src[y, x] != 0).any():
                    result[y, x] = img_src[y, x]

        return result

    def alpha_blending(self, src_img, tar_img):
        """
        Use certer weighting to do alpha blending.
        """
        h, w = tar_img.shape[0], tar_img.shape[1]

        mask_tar = cv2.inRange(tar_img, 0, 0)
        mask_src = cv2.inRange(src_img, 0, 0)
        mask_tar = cv2.medianBlur(mask_tar, 5) # there are some noise in the mask!
        mask_src = cv2.medianBlur(mask_src, 5)
        mask_tar = cv2.bitwise_not(mask_tar)
        mask_src = cv2.bitwise_not(mask_src)
        dis_tar = cv2.distanceTransform(mask_tar, cv2.DIST_L2, maskSize=5)
        dis_src = cv2.distanceTransform(mask_src, cv2.DIST_L2, maskSize=5)
        dis_tar = dis_tar / np.max(dis_tar)
        dis_src = dis_src / np.max(dis_src)

        #create_im_window("mask_src", mask_src)
        #create_im_window("mask_tar", mask_tar)

        # create the mask of the overlapping area
        mask_overlap = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                if (mask_tar[y, x] != 0 and mask_src[y, x] != 0):
                    mask_overlap[y, x] = 255
        
        weights_src = np.zeros((h, w))
        weights_tar = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                if mask_overlap[y, x] != 0:
                    weights_src[y, x] = dis_src[y, x] / (dis_src[y, x] + dis_tar[y, x])
                    weights_tar[y, x] = dis_tar[y, x] / (dis_src[y, x] + dis_tar[y, x])
                
                else:
                    if mask_tar[y ,x] != 0:
                        weights_tar[y, x] = 1
                    if mask_src[y, x] != 0:
                        weights_src[y, x] = 1
                
        #create_im_window("src", weights_src)
        #create_im_window("tar", weights_tar)

        scr_img = (src_img * weights_src.reshape((h, w, 1))).astype(np.uint8)
        tar_img = (tar_img * weights_tar.reshape((h, w, 1))).astype(np.uint8)
        result = (scr_img + tar_img)

        return result


if __name__ == '__main__':
    
    # baseline
    img1, img1_gray = read_img("./baseline/m1.jpg")
    img2, img2_gray = read_img("./baseline/m2.jpg")
    img3, img3_gray = read_img("./baseline/m3.jpg")
    img4, img4_gray = read_img("./baseline/m4.jpg")
    img5, img5_gray = read_img("./baseline/m5.jpg")
    img6, img6_gray = read_img("./baseline/m6.jpg")

    img_list = [img1, img2, img3, img4, img5, img6]

    img_tar = img_list[0]
    for i in range(1, len(img_list)):
        img_src = img_list[i]
        kp1, des1 = get_kp_des(img_to_gray(img_src))
        kp2, des2 = get_kp_des(img_to_gray(img_tar))
        
        match_points = knn_point_matcher(kp1, kp2, des1, des2, k=2, threshold=0.75)
        H = RANSAC(match_points, threshold=5)

        img_src, img_tar = warp_images(img_src, img_tar, H)

        blender = Blender()
        if i != 1:
            result_alpha_blend = blender.alpha_blending(img_src, img_tar)
            img_tar = result_alpha_blend
        else:
            result_dir_blend = blender.direct_blending(img_src, img_tar)
            result_alpha_blend = blender.alpha_blending(img_src, img_tar)
            cv2.imwrite("./result/dir_blend_2.jpg", result_dir_blend)
            cv2.imwrite("./result/alpha_blend_2.jpg", result_alpha_blend)
            
    cv2.imwrite("./result/alpha_blend_6.jpg", result_alpha_blend)

    # bonus
    img1, img1_gray = read_img("./bonus/m1.jpg")
    img2, img2_gray = read_img("./bonus/m2.jpg")
    img3, img3_gray = read_img("./bonus/m3.jpg")
    img4, img4_gray = read_img("./bonus/m4.jpg")

    img_list = [img4, img3, img2, img1]

    img_tar = img_list[0]
    for i in range(1, len(img_list)):
        img_src = img_list[i]
        kp1, des1 = get_kp_des(img_to_gray(img_src))
        kp2, des2 = get_kp_des(img_to_gray(img_tar))
        
        match_points = knn_point_matcher(kp1, kp2, des1, des2, k=2, threshold=0.75)
        H = RANSAC(match_points, threshold=5)

        img_src, img_tar = warp_images(img_src, img_tar, H)

        blender = Blender()
        result_alpha_blend = blender.alpha_blending(img_src, img_tar)
        img_tar = result_alpha_blend

    cv2.imwrite("./result/alpha_blend_bonus.jpg", result_alpha_blend)
