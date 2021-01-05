'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
from submission import *
import helper

def eight():
    data = np.load('../data/some_corresp.npz')
    #data = np.load('../data/some_corresp_noisy.npz')
    print(data)
    pts1 = data['pts1']
    pts2 = data['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(np.shape(im1))
    F = eightpoint(pts1, pts2, M)
    #RANSAC
    #F,_ = ransacF(pts1, pts2, M)
    helper.displayEpipolarF(im1,im2,F)
    
def findM2():
    pts = np.load('../data/some_corresp.npz')
    #pts = np.load('../data/some_corresp_noisy.npz')

    pts1 = pts["pts1"]
    pts2 = pts["pts2"]
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(im1.shape)
    F = eightpoint(pts1, pts2, M)
    #ransacF(pts1,pts2,M)
    intrinsic = np.load('../data/intrinsics.npz')
    K1, K2 = intrinsic['K1'], intrinsic['K2']
    E = essentialMatrix(F, K1, K2)
    #print('E ')
    #print(E)
    M2comb = helper.camera2(E)

    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    C1 = np.dot(K1, M1)

    for i in range(M2comb.shape[-1]):
        M2 = M2comb[:, :, i]
        C2 = np.dot(K2, M2)
        w, err = triangulate(C1, pts1, C2, pts2)
        if (np.all(w[:,2] > 0)):
            print('inside')
            print(i)
            break
    C2 = np.dot(K2, M2)
    np.savez('q3_3.npz', M2 = M2, C2 = C2, P = w)
    return M1,C1,M2,C2,F
if __name__ == "__main__":
    #eight()
    findM2()

    