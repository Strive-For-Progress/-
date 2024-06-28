import cv2
import numpy as np


def rgb_to_hsi(rgb_image):
    height, width = rgb_image.shape[:2]
    hsi_image = np.zeros_like(rgb_image, dtype=float)

    B = rgb_image[:,:,0] / 255.0
    G = rgb_image[:,:,1] / 255.0
    R = rgb_image[:,:,2] / 255.0

    for y in range(height):
        for x in range(width):
            b = B[y, x]
            g = G[y, x]
            r = R[y, x]

            intensity = (r + g + b) / 3

            num = (2 * r - g - b) * 0.5
            den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) 
            theta = np.arccos(num / (den + 1e-6))

            if b <= g:
                hue = theta
            else:
                hue = 2 * np.pi - theta

            saturation = 1.0 - (3.0 * min(r, g, b) / (r + g + b + 1e-6))

            hsi_image[y, x, 0] = hue / (2*np.pi) * 255
            hsi_image[y, x, 1] = saturation  * 255
            hsi_image[y, x, 2] = intensity * 255

    return hsi_image

def hsi_to_rgb(hsi_image):
    # Split HSI channels  & Normalize H, S, I channels
    H = hsi_image[:,:,0] / 255.0
    S = hsi_image[:,:,1] / 255.0
    I = hsi_image[:,:,2] / 255.0

    height, width = H.shape
    rgb_image =  np.zeros_like(hsi_image, dtype=float)


    for y in range(height):
        for x in range(width):
            hue = H[y, x] * 2 * np.pi
            saturation = S[y, x]
            intensity = I[y, x]

            if saturation == 0:
                r = g = b = intensity
            else:
                if 0 <= hue < 2 * np.pi / 3:
                    b = intensity * (1 - saturation)
                    r = intensity * (1 + (saturation * np.cos(hue)) / np.cos(np.pi / 3 - hue))
                    g = 3 * intensity - (r + b)
                elif 2 * np.pi / 3 <= hue < 4 * np.pi / 3:
                    hue -= 2 * np.pi / 3
                    r = intensity * (1 - saturation)
                    g = intensity * (1 + (saturation * np.cos(hue)) / np.cos(np.pi / 3 - hue))
                    b = 3 * intensity - (r + g)
                elif 4 * np.pi / 3 <= hue < 2 * np.pi:
                    hue -= 4 * np.pi / 3
                    g = intensity * (1 - saturation)
                    b = intensity * (1 + (saturation * np.cos(hue)) / np.cos(np.pi / 3 - hue))
                    r = 3 * intensity - (g + b)

            rgb_image[y, x, 2] = r * 255
            rgb_image[y, x, 1] = g * 255
            rgb_image[y, x, 0] = b * 255

    return rgb_image.astype(np.uint8)


def rgb_to_lab(image_bgr):

    def f(x): 
        if x > 0.008856:
            return np.power(x, 1/3)
        else:
            return (7.787 * x) + (16.0/116.0)

    # 將XYZ色彩空間轉換為Lab色彩空間
    image_lab = np.zeros_like(image_bgr)

    for i in range(image_bgr.shape[0]):
        for j in range(image_bgr.shape[1]):
            # 將影像從0-255範圍轉換為0-1範圍
            r = image_bgr[i][j][2]/ 255.0
            g = image_bgr[i][j][1]/ 255.0
            b = image_bgr[i][j][0]/ 255.0
            
            x = (0.412453*r + 0.357580*g +  0.180423*b) / 0.950456
            y =  0.212671*r + 0.715160*g +  0.072169*b
            z = (0.019334*r + 0.119193*g +  0.950227*b) / 1.088754

            fx = f(x)
            fy = f(y)
            fz = f(z)

            l = 116 * fy - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)

            [image_lab[i,j,0], image_lab[i,j,1], image_lab[i,j,2]] = np.clip([l, a+128, b+128], 0, 225)

    return image_lab

def lab_to_bgr(image_lab):

    def f_1(x): 
        if x > 6/29:
            return np.power(x, 3)
        else:
            return (x - 4.0/29.0) * (108.0/841.0)

    rgb_image = np.zeros_like(image_lab, dtype=float)

    for i in range(image_lab.shape[0]):
        for j in range(image_lab.shape[1]):
            # 將影像從0-100範圍轉換為0-1範圍
            l = image_lab[i][j][0]
            a = image_lab[i][j][1] - 128
            b = image_lab[i][j][2] - 128

            # 將影像從Lab轉換為XYZ色彩空間
            fy = (l + 16) / 116
            fx = fy + a / 500 
            fz = fy - b / 200

            x = f_1(fx) * 0.950456
            y = f_1(fy) 
            z = f_1(fz) * 1.088754
            
            R =  (3.240479*x  + (-1.537150)*y + (-0.498535)*z) 
            G =  ((-0.969256)*x  + 1.875992*y  + 0.041556*z) 
            B =  (0.055648*x  + (-0.204043)*y + 1.057311*z) 

            [R, G, B] = np.clip([R,G,B],0,1)

            rgb_image[i, j, 2] = R*255
            rgb_image[i, j, 1] = G*255
            rgb_image[i, j, 0] = B*255

    return rgb_image.astype(np.uint8)



def gamma_correction_rgb(image, gamma_value):
    image = image.astype(np.float32) / 255.0
    
    corrected_image = np.power(image, gamma_value)
    # 将像素值范围缩放回0到255
    corrected_image = (corrected_image * 255).astype(np.uint8)
    return corrected_image

def gamma_correction_hsi(image_hsi, gamma):
    # Split HSI channels
    H, S, I = cv2.split(image_hsi)
    
    # Apply gamma correction to the intensity channel
    corrected_I = np.power(I/255.0, gamma) * 255.0

    # Merge the corrected intensity channel back into the HSI image
    corrected_hsi_img = cv2.merge([H, S, corrected_I])

    return corrected_hsi_img

def gamma_correction_lab(image_lab, gamma):
    # Split Lab channels
    L, a, b = cv2.split(image_lab)

    # Apply gamma correction to the L channel
    corrected_L = np.power(L/100.0, gamma) * 100.0
    corrected_L = corrected_L.astype(np.uint8)

    # Merge the corrected L channel back into the Lab image
    corrected_lab_img = cv2.merge([corrected_L, a, b])

    return corrected_lab_img