import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.signal import medfilt2d

image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image


# get the normal vector of a certain pixel
def get_norm_vec(I, S): 
    b = np.linalg.inv(S.T @ S) @ S.T @ I
    vec_norm = np.linalg.norm(b)
    if vec_norm != 0:
        b = b / vec_norm
    return b

# get the normal map of a given object, object is a string
def get_norm_map(object):
    img_1 = read_bmp("./test/"+object+"/pic1.bmp")
    img_2 = read_bmp("./test/"+object+"/pic2.bmp")
    img_3 = read_bmp("./test/"+object+"/pic3.bmp")
    img_4 = read_bmp("./test/"+object+"/pic4.bmp")
    img_5 = read_bmp("./test/"+object+"/pic5.bmp")
    img_6 = read_bmp("./test/"+object+"/pic6.bmp")
    txt = open("./test/"+object+"/LightSource.txt", "r")

    img_shape = img_1.shape
    
    light_matrix = []
    for line in txt.readlines():
        light_source = line.split()[1][1:-1].split(",")
        light_source = [int(s) for s in light_source]
        light_matrix.append(light_source)
    light_matrix = np.array(light_matrix)
    #print(light_matrix)

    norm_map = np.empty((img_shape[0], img_shape[1], 3))
    for y in range(img_shape[0]):
        for x in range(img_shape[1]):
            intesity_vec = np.array([img_1[y, x], img_2[y, x], img_3[y, x], 
                                    img_4[y, x], img_5[y, x], img_6[y, x]])
            norm_vec = get_norm_vec(intesity_vec, light_matrix / np.linalg.norm(light_matrix, axis=1, keepdims=True))
            norm_map[y, x] = norm_vec
    return norm_map

# get the mask given the normal map
def get_mask(norm_map):
    img_shape = norm_map.shape[:2]
    mask = np.zeros((img_shape[0], img_shape[1]))
    for y in range(img_shape[0]):
        for x in range(img_shape[1]):
            if norm_map[y, x, 2] != 0: mask[y, x] = 1
    return mask

# get depth map given the normal map
def get_depth_map(norm_map):
    mask = get_mask(norm_map)
    img_shape = norm_map.shape[:2]

    num_matrix = np.zeros((img_shape[0], img_shape[1]), dtype=int)
    num_pixel = 1
    for y in range(img_shape[0]):
        for x in range(img_shape[1]):
            if mask[y, x] != 0: 
                num_matrix[y, x] = num_pixel
                num_pixel += 1
    obj_size = num_pixel - 1

    M = np.zeros((2*obj_size, obj_size))
    V = np.zeros((2*obj_size))
    for n in range(1, obj_size+1):
        idx = n - 1
        coord = np.where(num_matrix == n)
        irow = int(coord[0])
        icol = int(coord[1])
        nx = norm_map[irow, icol, 0]
        ny = norm_map[irow, icol, 1]
        nz = norm_map[irow, icol, 2]
        
        if num_matrix[irow, icol+1] > 0 and num_matrix[irow-1, icol] > 0:
            V[2*idx] = nx / nz
            M[2*idx, num_matrix[irow, icol]-1] = 1
            M[2*idx, num_matrix[irow, icol+1]-1] = -1
            V[2*idx+1] = ny / nz
            M[2*idx+1, num_matrix[irow, icol]-1] = 1
            M[2*idx+1, num_matrix[irow-1, icol]-1] = -1    
        elif num_matrix[irow, icol+1] == 0 and num_matrix[irow-1, icol] > 0:
            if num_matrix[irow, icol-1] > 0:
                V[2*idx] = nx / nz
                M[2*idx, num_matrix[irow, icol]-1] = 1
                M[2*idx, num_matrix[irow, icol-1]-1] = -1
            V[2*idx+1] = ny / nz
            M[2*idx+1, num_matrix[irow, icol]-1] = 1
            M[2*idx+1, num_matrix[irow-1, icol]-1] = -1
        elif num_matrix[irow, icol+1] > 0 and num_matrix[irow-1, icol] == 0:
            V[2*idx] = nx / nz
            M[2*idx, num_matrix[irow, icol]-1] = 1
            M[2*idx, num_matrix[irow, icol+1]-1] = -1
            if num_matrix[irow+1, icol] > 0:
                V[2*idx+1] = ny / nz
                M[2*idx+1, num_matrix[irow, icol]-1] = 1
                M[2*idx+1, num_matrix[irow+1, icol]-1] = -1
        else:
            if num_matrix[irow, icol-1] > 0:
                V[2*idx] = nx / nz
                M[2*idx, num_matrix[irow, icol]-1] = 1
                M[2*idx, num_matrix[irow, icol-1]-1] = -1
            if num_matrix[irow+1, icol] > 0:
                V[2*idx+1] = ny / nz
                M[2*idx+1, num_matrix[irow, icol]-1] = 1
                M[2*idx+1, num_matrix[irow+1, icol]-1] = -1
                
    M = sparse.csr_matrix(M)
    z_vec = sparse.linalg.inv(M.T @ M) @ M.T @ V
    Z = np.zeros((img_shape[0], img_shape[1]))
    for i in range(obj_size):
        num = i + 1
        coord = np.where(num_matrix == num)
        irow = int(coord[0])
        icol = int(coord[1])
        Z[irow, icol] = z_vec[i]
    return Z


if __name__ == '__main__':
    
    bunny_norm = get_norm_map("bunny")
    normal_visualization(bunny_norm)
    bunny_depth = get_depth_map(bunny_norm)
    depth_visualization(bunny_depth)
    save_ply(bunny_depth, "./result/bunny.ply")

    star_norm = get_norm_map("star")
    normal_visualization(star_norm)
    star_depth = get_depth_map(star_norm)
    depth_visualization(star_depth)
    save_ply(star_depth, "./result/star.ply")

    venus_norm = get_norm_map("venus")
    normal_visualization(venus_norm)
    venus_depth = get_depth_map(venus_norm)
    depth_visualization(venus_depth)
    save_ply(venus_depth, "./result/venus_presmoothed.ply")
    smoothed_depth = medfilt2d(venus_depth, kernel_size=3)
    depth_visualization(smoothed_depth)
    save_ply(smoothed_depth, "./result/venus.ply")

    # showing the windows of all visualization function
    plt.show()