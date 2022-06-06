import cv2 as cv
import numpy as np
import easyocr

from pathlib import Path
from matplotlib import pyplot

input_filename = "numberplate5.png"
output_filename = "output_images/extension.png"

SHOW_DEBUG_FIGURES = True

def main():
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
    (x_range, y_range) = np.where(mask > 0)
    min_x, min_y = min(x_range), min(y_range)
    max_x, max_y = max(x_range), max(y_range)
    
    # Crop out the license plate.
    license_cropped = image_grayscale[min_x:max_x, min_y:max_y]
    pyplot.imshow(cv.cvtColor(license_cropped, cv.COLOR_BGR2RGB))

    # Using OCR, extract and build text from license plate.
    reader = easyocr.Reader(['en'])
    result = reader.readtext(license_cropped, detail=0)
    
    text = ""
    for result in result:
        text = text + " " + result
    
    print("License plate number:" + text)

    cv.imwrite("output_images/extension.png", image_masked)

    if SHOW_DEBUG_FIGURES:
        pyplot.show()


if __name__ == "__main__":
    main()
