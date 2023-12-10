import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def display_image_with_matplotlib(image, title=None, figsize=(8, 8), cmap=None, axis_off=True):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=cmap)
    if axis_off:
        ax.axis('off')
    if title:
        ax.set_title(title)
    plt.show()

def display_image_with_cv2(image,title):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_image_histogram(image):
    pd.Series(image.flatten()).plot(kind='hist')
    plt.show()

def display_rgb_channels(image, title=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, color in enumerate(['Reds', 'Greens', 'Blues']):
        axs[i].imshow(image[:,:,i], cmap=color)
        axs[i].set_title(f'{color[:-1]} channel')
    plt.show()

def display_images_cv2_vs_matplotlib(cv2_image, matplotlib_image):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2_image)
    axs[1].imshow(matplotlib_image)
    axs[0].set_title('CV2 Image')
    axs[1].set_title('Matplotlib Image')
    plt.show()

def resize_and_display_image(image, target_size=None, interpolation=cv2.INTER_LINEAR):
    if target_size:
        resized_image = cv2.resize(image, target_size, interpolation=interpolation)
    else:
        resized_image = image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(resized_image)
    ax.axis('off')
    plt.show()

image_plt= plt.imread('party.jpg')
image_cv2= cv2.imread('party.jpg')
image_cv2=cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

display_image_with_matplotlib(image_plt, title='Matplotlib Image')
display_image_with_cv2(image_cv2,title='CV2 Image')
plot_image_histogram(image_cv2)
display_images_cv2_vs_matplotlib(image_cv2, image_plt)
resize_and_display_image(image_cv2, target_size=(200, 200))
resize_and_display_image(image_cv2, target_size=(1500, 2000), interpolation=cv2.INTER_CUBIC)
