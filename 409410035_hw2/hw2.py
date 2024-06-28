import cv2
import numpy as np

def apply_filter(image, kernel):

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    pad_top = pad_bottom = kernel_height // 2
    pad_left = pad_right = kernel_width // 2
    
    # Pad the input image with zeros
    padded_image = np.zeros((image_height + pad_top + pad_bottom, image_width + pad_left + pad_right))
    padded_image[pad_top : (image_height + pad_top), pad_left : (image_width + pad_left)] = image
    
    # Create an empty output image
    output_image = np.zeros_like(image, dtype=image.dtype)
    
    for row in range(pad_top, image_height + pad_top):
        for col in range(pad_left, image_width + pad_left):
            # Extract the region of the image centered at the current pixel, with the same size as the kernel
            region = padded_image[row-pad_top:row+pad_top+1, col-pad_left:col+pad_left+1]
            
            # Apply the kernel to the region of the image & Store the output pixel in the output image
            output_image[row-pad_top, col-pad_left] = (region * kernel).sum()
    
    return output_image

if __name__ =="__main__":
    # Load the two gray-level images
    img1 = cv2.imread('./HW2_test_image/blurry_moon.tif',   cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./HW2_test_image/skeleton_orig.bmp', cv2.IMREAD_GRAYSCALE)

    # Define the Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Apply Laplacian operator to the images
    lap_img1 = img1 - apply_filter(img1, laplacian_kernel)
    lap_img2 = img2 - apply_filter(img2, laplacian_kernel)
    lap_img1 = np.clip(lap_img1, 0, 255).astype(np.uint8)
    lap_img2 = np.clip(lap_img2, 0, 255).astype(np.uint8)


    boost_factor = 2.3
    # Apply high-boost filtering to the images
    boost_img1 = (boost_factor-1) * img1 +  apply_filter(img1, laplacian_kernel)
    boost_img2 = (boost_factor-1) * img2 +  apply_filter(img2, laplacian_kernel)
    boost_img1 = np.clip(boost_img1, 0, 255).astype(np.uint8)
    boost_img2 = np.clip(boost_img2, 0, 255).astype(np.uint8)

    # Display the original images and the sharpened images side by side
    cv2.imshow('Original Image 1', img1)
    cv2.imshow('Laplacian Sharpened Image 1', lap_img1)
    cv2.imshow('High-Boost Sharpened Image 1', boost_img1)

    cv2.imshow('Original Image 2', img2)
    cv2.imshow('Laplacian Sharpened Image 2', lap_img2)
    cv2.imshow('High-Boost Sharpened Image 2', boost_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()