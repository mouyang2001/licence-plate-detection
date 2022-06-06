import cv2 as cv
import numpy as np
import easyocr
import sys

from pathlib import Path
from matplotlib import pyplot
from matplotlib.patches import Rectangle

def main():

    # Commmand line interface code.
    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True
    input_filename = "numberplate6.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output_extension.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])



    # Image processing with OpenCV.
    image_color = cv.imread(input_filename)
    image_grayscale = cv.cvtColor(image_color, cv.COLOR_BGR2GRAY)

    image_blur = cv.bilateralFilter(image_grayscale, 7, 19, 19)
    image_canny = cv.Canny(image_blur, 40, 200)

    contours, hierarchy = cv.findContours(image_canny.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    # Find contour that approximates a square and has the largest area.
    license_polygon = None
    for contour in contours:
        approximate_polygon = cv.approxPolyDP(contour, 10, True)
        if len(approximate_polygon) == 4:
            license_polygon = approximate_polygon
            break



    # Mask out everything except the license plate.
    mask = np.zeros(image_grayscale.shape, np.uint8)
    image_masked = cv.drawContours(mask, [license_polygon], 0, 255, -1)
    image_masked = cv.bitwise_and(image_color, image_color, mask=mask)

    # Using mask calculated via drawContour, find bounding box.
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = 0, 0, 0, 0
    (x_range, y_range) = np.where(mask > 0)
    x_min, y_min = min(x_range), min(y_range)
    x_max, y_max = max(x_range), max(y_range)
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = y_min, y_max, x_min, x_max
    
    # Crop out the license plate.
    license_cropped = image_grayscale[x_min:x_max, y_min:y_max]

    # Using OCR, extract and build text from license plate.
    reader = easyocr.Reader(['en'])
    result = reader.readtext(license_cropped, detail=0)
    text = ""
    for result in result:
        text = text + " " + result
    


    # Result and pyplot debug figure.
    print("License plate number:" + text)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('#1 Grayscale')
    axs1[0, 0].imshow(image_grayscale, cmap='gray')
    axs1[0, 1].set_title('#2 Bilateral filter')
    axs1[0, 1].imshow(image_blur, cmap='gray')
    axs1[1, 0].set_title('#3 Edge detection')
    axs1[1, 0].imshow(image_canny, cmap='gray')
    axs1[1, 1].set_title('#4 License plate')
    axs1[1, 1].imshow(image_grayscale, cmap='gray')

    # draw bounding box
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, 
                    linewidth=3, edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        pyplot.show()


if __name__ == "__main__":
    main()
