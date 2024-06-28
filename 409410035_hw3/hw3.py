import cv2
import numpy as np
import unity


# 處理整體太暗的圖片（aloe.jpg 和 church.jpg）
def adjust_dark_image(image, gamma_value=0.7):


    # Apply gamma correction to the HSI image
    hsi = unity.rgb_to_hsi(image)
    adjusted_hsi = unity.gamma_correction_hsi(hsi, gamma_value)
    result_hsi = unity.hsi_to_rgb(adjusted_hsi)

    # LAB色彩空间
    lab = unity.rgb_to_lab(image)
    adjusted_lab = unity.gamma_correction_lab(lab, gamma_value)
    result_lab = unity.lab_to_bgr(adjusted_lab)

    # RGB色彩空间
    adjusted_rgb = unity.gamma_correction_rgb(image, gamma_value)

    return adjusted_rgb, result_hsi, result_lab



# 處理局部太亮的圖片（house.jpg 和 kitchen.jpg）
def adjust_bright_image(image, gamma_value=1.5):

    # Apply gamma correction to the HSI image
    hsi = unity.rgb_to_hsi(image)
    adjusted_hsi = unity.gamma_correction_hsi(hsi, gamma_value)
    result_hsi = unity.hsi_to_rgb(adjusted_hsi)

    # LAB色彩空间
    lab = unity.rgb_to_lab(image)
    adjusted_lab = unity.gamma_correction_lab(lab, gamma_value)
    result_lab = unity.lab_to_bgr(adjusted_lab)

    # RGB色彩空间
    adjusted_rgb = unity.gamma_correction_rgb(image, gamma_value)

    return adjusted_rgb, result_hsi, result_lab



if __name__ == "__main__":

    # 載入圖片
    aloe_image = cv2.imread('aloe.jpg')
    church_image = cv2.imread('church.jpg')
    house_image = cv2.imread('house.jpg')
    kitchen_image = cv2.imread('kitchen.jpg')


    adjusted_aloe_rgb, adjusted_aloe_hsi, adjusted_aloe_lab = adjust_dark_image(aloe_image, gamma_value=0.7)
    adjusted_church_rgb, adjusted_church_hsi, adjusted_church_lab = adjust_dark_image(church_image, gamma_value=0.7)


    # 顯示處理後的圖片
    cv2.imshow('Aloe', aloe_image)
    cv2.imshow('Adjusted Aloe (RGB)', adjusted_aloe_rgb)
    cv2.imshow('Adjusted Aloe (HSI)', adjusted_aloe_hsi)
    cv2.imshow('Adjusted Aloe (LAB)', adjusted_aloe_lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Church', church_image)
    cv2.imshow('Adjusted Church (RGB)', adjusted_church_rgb)
    cv2.imshow('Adjusted Church (HSI)', adjusted_church_hsi)
    cv2.imshow('Adjusted Church (LAB)', adjusted_church_lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 處理局部太亮的圖片
    adjusted_house_rgb, adjusted_house_hsi, adjusted_house_lab = adjust_bright_image(house_image, gamma_value=1.5)
    adjusted_kitchen_rgb, adjusted_kitchen_hsi, adjusted_kitchen_lab = adjust_bright_image(kitchen_image, gamma_value=1.5)


    # 顯示原圖 & 處理後的圖片

    cv2.imshow('house', house_image)
    cv2.imshow('Adjusted house (RGB)', adjusted_house_rgb)
    cv2.imshow('Adjusted house (HSI)', adjusted_house_hsi)
    cv2.imshow('Adjusted house (LAB)', adjusted_house_lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('kitchen', kitchen_image)
    cv2.imshow('Adjusted kitchen (RGB)', adjusted_kitchen_rgb)
    cv2.imshow('Adjusted kitchen (HSI)', adjusted_kitchen_hsi)
    cv2.imshow('Adjusted kitchen (LAB)', adjusted_kitchen_lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
