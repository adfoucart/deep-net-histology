from matplotlib import pyplot as plt
import numpy as np

def view_firstLevel_kernels(W):
    n = W.shape[3]
    rows = np.ceil(np.sqrt(n))
    cols = np.round(np.sqrt(n))

    plt.figure()
    for i in range(n):
        W_ = W[:,:,:,i]
        W_ = (W_ - W_.min())/(W_.max()-W_.min())
        plt.subplot(rows, cols, i+1)
        plt.imshow(W_)
        plt.axis('off')
    plt.show()

def view_secondLevel_kernels(W2, W1):
    n2 = W2.shape[3]
    n1 = W2.shape[2]
    rows = np.ceil(np.sqrt(n2))
    cols = np.round(np.sqrt(n2))

    plt.figure()
    for i in range(n2):
        W2_ = W2[:,:,:,i]
        W2_1 = np.zeros((W2_.shape[0], W2_.shape[1], 3))
        for j in range(n1):
            W1_ = W1[:,:,:,j]
            for c in range(3):
                W2_1[:,:,c] += W1_[:,:,c]*W2_[:,:,j]
        W2_1 = (W2_1 - W2_1.min())/(W2_1.max()-W2_1.min())
        plt.subplot(rows, cols, i+1)
        plt.imshow(W2_1)
        # print W2_1.mean()
        plt.axis('off')
    plt.show()