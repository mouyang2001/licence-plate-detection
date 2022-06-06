import cv2 as cv

from pathlib import Path
from matplotlib import pyplot

input_filename = "numberplate5.png"
output_filename = "output_images/extension.png"

def main():
    image_color = cv.imread(input_filename)
    image_grayscale = cv.cvtColor(image_color, cv.COLOR_BGR2GRAY)

    # Use of bilateral filter instead of guassian. Slower but better at removing noise and preserving edges.
    # cv.bilateralFilter(image_greyscale, filter_size, sigmaColor, sigmaSpace)
    image_blur = cv.bilateralFilter(image_grayscale, 5, 20, 20)

    # Canny edge detection.
    image_edge = cv.Canny(image_blur, 60, 200)

    # Contour retrival and sort by area.
    contours, hierarchy = cv.findContours(image_edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    


    # Plot steps to final output.
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Grayscale')
    axs1[0, 0].imshow(image_grayscale, cmap='gray')
    axs1[0, 1].set_title('Bilateral Filter')
    axs1[0, 1].imshow(image_blur, cmap='gray')
    axs1[1, 0].set_title('Threshold')
    axs1[1, 0].imshow(image_edge, cmap='gray')
    axs1[1, 1].set_title('Edges')
    axs1[1, 1].imshow(contours, cmap='gray')

    # Saves output image.
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    # GUI display output.
    pyplot.show()

    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
