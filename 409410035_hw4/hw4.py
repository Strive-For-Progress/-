import cv2
import numpy as np
import matplotlib.pyplot as plt


def median_blur(image, ksize=7):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Pad the image(上下左右各 ksize//2 像素) to handle borders
    padded_image = np.pad(image, ((ksize//2, ksize//2), (ksize//2, ksize//2)), mode='constant')
    
    # Create a blank output image
    output_image = np.zeros_like(image)
    
    # Apply median blur
    for i in range(height):
        for j in range(width):
            # patch 是指 image 中以 (i,j) 為中心的 ksize x ksize 區域
            patch = padded_image[i:i+ksize, j:j+ksize]
            median_value = np.median(patch)
            output_image[i, j] = median_value
    
    return output_image


def sobel_operator(image, axis):
    # Define Sobel kernels
    if axis == 'x':
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif axis == 'y':
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Perform convolution
    filtered_image = np.zeros_like(image)
    height, width = image.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            # patch 是指 image 中以 (i,j) 為中心的 3x3 區域
            patch = image[i-1:i+2, j-1:j+2]
            filtered_value = np.sum(patch * kernel)
            filtered_image[i, j] = filtered_value
    
    return filtered_image



def color_edge_detection(image, apply_blur):
    # Convert image to grayscale
    gray = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    
    if apply_blur.lower() == 'y':
        # Apply median blur
        blur_ksize = 7
        blurred_image = median_blur(gray, blur_ksize)
    else:
        blurred_image = gray

    # Apply Sobel operator for x-axis gradient
    sobel_x = sobel_operator(blurred_image, 'x')

    # Apply Sobel operator for y-axis gradient
    sobel_y = sobel_operator(blurred_image, 'y')
    
    # Compute gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # direction = np.arctan2(sobel_y, sobel_x)
    
    return magnitude


def bgr_to_rgb(image):
    # Swap the channels from BGR to RGB
    rgb_image = image[:, :, ::-1]
    return rgb_image


if __name__=='__main__':
    # Load color images
    image1 = cv2.imread('baboon.png')
    image2 = cv2.imread('peppers.png')
    image3 = cv2.imread('pool.png')

    apply_blur = input("Do you want to apply median blur (filter size is 7x7)? (y/n): ")
    
  
    # Perform edge detection on the color images
    edges1 = color_edge_detection(image1, apply_blur)
    edges2 = color_edge_detection(image2, apply_blur)
    edges3 = color_edge_detection(image3, apply_blur)

    # Display original images and processed images
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Show original Image 1
    axs[0, 0].imshow(bgr_to_rgb(image1))
    axs[0, 0].set_title('Image 1')
    axs[0, 0].axis('off')

    # Show original Image 2
    axs[0, 1].imshow(bgr_to_rgb(image2))
    axs[0, 1].set_title('Image 2')
    axs[0, 1].axis('off')

    # Show original Image 3
    axs[0, 2].imshow(bgr_to_rgb(image3))
    axs[0, 2].set_title('Image 3')
    axs[0, 2].axis('off')

    # Show edges of Image 1
    axs[1, 0].imshow(edges1, cmap='gray')
    axs[1, 0].set_title('Edges 1')
    axs[1, 0].axis('off')

    # Show edges of Image 2
    axs[1, 1].imshow(edges2, cmap='gray')
    axs[1, 1].set_title('Edges 2')
    axs[1, 1].axis('off')

    # Show edges of Image 3
    axs[1, 2].imshow(edges3, cmap='gray')
    axs[1, 2].set_title('Edges 3')
    axs[1, 2].axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()