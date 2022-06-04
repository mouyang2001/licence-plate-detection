import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array



# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate1.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here

    # Compute greyscale image from RGB image.
    print("Computing greyscale image...")
    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    # Compute std_dev image then stretch min-max scaling 0 to 255.
    # TODO: Potentially swap around.
    print("Computing standard deviation image...")
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    print("Computing 0 to 255 stretched image...")
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)
    
    # Compute threshold image with simple thresholding.
    # TODO: Adaptive thresholding.
    px_array = computeThresholdGE(px_array, 150, image_width, image_height)

    # Compute dialation and erosion.
    for i in range(7):
        print("Computing dialation step #{}...".format(i+1))
        px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    
    for i in range(7):
        print("Computing erosion step #{}...".format(i+1))
        px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

    print("Processing complete!")

    # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    bbox_min_x = center_x - image_width / 4.0
    bbox_max_x = center_x + image_width / 4.0
    bbox_min_y = center_y - image_height / 4.0
    bbox_max_y = center_y + image_height / 4.0





    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):

    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(image_height):
        for col in range(image_width):
            r = pixel_array_r[row][col]
            g = pixel_array_g[row][col]
            b = pixel_array_b[row][col]
            greyscale_pixel_array[row][col] = round(
                0.299 * r + 0.587 * g + 0.114 * b)

    return greyscale_pixel_array


def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(2, image_height-2):
        for c in range(2, image_width-2):
            nums = []
            for i in [-2, -1, 0, 1, 2]:
                for j in [-2, -1, 0, 1, 2]:
                    nums.append(pixel_array[r+i][c+j])

            image[r][c] = get_standard_deviation(nums)

    return image


def get_standard_deviation(nums):
    length = len(nums)
    mean = sum(nums) / length
    variance = sum((x - mean)**2 for x in nums) / length
    standard_deviation = variance ** 0.5
    return standard_deviation


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    minimum, maximum = 255, 0

    for r in range(image_height):
        for c in range(image_width):
            minimum = min(minimum, pixel_array[r][c])
            maximum = max(maximum, pixel_array[r][c])

    return (minimum, maximum)


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)

    (fmin, fmax) = computeMinAndMaxValues(pixel_array, image_width, image_height)
    a = 1 if fmax == fmin else 255 / (fmax - fmin)
    b = a - fmin

    for r in range(image_height):
        for c in range(image_width):
            image[r][c] = round(a * pixel_array[r][c] + b)

    return image


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)
    for r in range(image_height):
        for c in range(image_width):
            image[r][c] = 1 if pixel_array[r][c] >= threshold_value else 0

    return image

# TODO: Potentially optimize border handling.
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    dilation = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(image_height):
        for c in range(image_width):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if (r+i >= 0 and r+i < image_height and 
                        c+j >= 0 and c+j < image_width and 
                        pixel_array[r+i][c+j] != 0):
                        dilation[r][c] = 1
                        break

    return dilation


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    erosion = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(image_height):
        for c in range(image_width):
            if r == 0 or r == image_height-1 or c == 0 or c == image_width-1:
                continue

            pixel_value = 1
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if pixel_array[r+i][c+j] == 0:
                        pixel_value = 0
                        break

            erosion[r][c] = pixel_value

    return erosion


if __name__ == "__main__":
    main()