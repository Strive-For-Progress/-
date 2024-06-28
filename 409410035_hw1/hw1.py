import matplotlib.pyplot as plt
import numpy as np
import cv2



def ProcessHist(img):
    pmf = np.zeros((256,),dtype=np.float16)
    cdf = np.zeros((256,),dtype=np.float16)
    T = np.zeros((256,),dtype=np.uint8)
    m,n = img.shape

    #comput pmf in case loss data ，muliple precent later
    for i in range(m):
        for j in range(n):
            pmf[img[i,j]] += 1

    precent = 1.0/(n*m)

    # compute cdf 
    for i in range(256):
        if (i == 0): cdf[i] = pmf[i] * precent
        else: cdf[i] =  cdf[i-1] + pmf[i] * precent
        T[i] = round(cdf[i] * 255)

    #apply transfrom function
    for i in range(m):
        for j in range(n):
            img[i,j]= T[img[i,j]]
    return img

def Global(path):
    img    = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    before = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = ProcessHist(img)
    Display (before, img)
    return 

def Display(beforeImg, afterImg):
    # Create a figure with subplots to display the images and histograms
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Plot the first image
    axs[0][0].imshow(cv2.cvtColor(beforeImg, cv2.COLOR_BGR2RGB))
    axs[0][0].set_title('Original Image')

    # Compute and plot the histogram in the second subplot
    hist, bins = np.histogram(beforeImg.ravel(), 256, [0, 256])
    axs[0][1].plot(hist)
    axs[0][1].set_title('Histogram')
    axs[0][1].set_xlim([0, 256])
    axs[0][1].set_ylim([0, 10000])
    

    # Plot the third image
    axs[1][0].imshow(cv2.cvtColor(afterImg, cv2.COLOR_BGR2RGB))
    axs[1][0].set_title('After Proccess Image')

    # Compute and plot the histogram in the forth subplot
    hist, bins = np.histogram(afterImg.ravel(), 256, [0, 256])
    axs[1][1].plot(hist)
    axs[1][1].set_title('Histogram')
    axs[1][1].set_xlim([0, 256])

    plt.show()
    return

def Slice(path):
    # Read the input image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Determine the dimensions of the image
    height, width = img.shape

    # Compute the size of each subimage
    sub_height = height // 4
    sub_width = width // 4

    # Slice the image into 16 subimages
    subimages = []
    for i in range(4):
        for j in range(4):
            subimage = img[i*sub_height:(i+1)*sub_height, j*sub_width:(j+1)*sub_width]
            subimages.append(subimage)

    
    # Return the subimages array
    return subimages

def Reconstruct(subimages, beforeImg): 
    height, width = beforeImg.shape
    reconstructed_img = np.zeros((height, width), dtype=np.uint8)
    sub_height = height // 4
    sub_width = width // 4

    # Copy the subimages back into the reconstructed image array
    for i in range(4):
        for j in range(4):
            subimage = subimages[i*4+j]
            reconstructed_img[i*sub_height:(i+1)*sub_height, j*sub_width:(j+1)*sub_width] = subimage

    return reconstructed_img

def Local(path):
    subimages = Slice(path)
    origin = Slice(path)

    # if want to know the final result only, you can disable line 111
    for i in range(16):
        afterImg = ProcessHist(subimages[i])
        Display(origin[i], afterImg)

    before = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    after = Reconstruct(subimages, before)

    Display(before, after)
    return

if __name__ =="__main__":
    Global('Lena.bmp')
    Local('Lena.bmp')
    Global('Peppers.bmp')
    Local('Peppers.bmp')
    