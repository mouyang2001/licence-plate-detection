def computeHistogram(pixel_array, image_width, image_height, nr_bins):
    res = [0.0 for x in range(nr_bins)]

    for r in range(image_height):
        for c in range(image_width):
            res[pixel_array[r][c]] += 1.0

    return res


def computeCumulativeHistogram(pixel_array, image_width, image_height, nr_bins):
    histogram = [0.0 for x in range(nr_bins)]
    for r in range(image_height):
        for c in range(image_width):
            histogram[pixel_array[r][c]] += 1.0

    cumulative = [0.0 for x in range(nr_bins)]
    cumulative[0] = histogram[0]
    for x in range(1, nr_bins):
        cumulative[x] = cumulative[x-1] + histogram[x]

    return cumulative


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    res = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(image_height):
        for c in range(image_width):
            if (pixel_array[r][c] >= threshold_value):
                res[r][c] = 255
            else:
                res[r][c] = 0

    return res


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):

    greyscale_pixel_array = createInitializedGreyscalePixelArray(
        image_width, image_height)

    # STUDENT CODE HERE
    for row in range(image_height):
        for col in range(image_width):
            r = pixel_array_r[row][col]
            g = pixel_array_g[row][col]
            b = pixel_array_b[row][col]
            greyscale_pixel_array[row][col] = round(
                0.299 * r + 0.587 * g + 0.114 * b)

    return greyscale_pixel_array


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    minimum, maximum = 255, 0

    for r in range(image_height):
        for c in range(image_width):
            minimum = min(minimum, pixel_array[r][c])
            maximum = max(maximum, pixel_array[r][c])

    return (minimum, maximum)


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    (fmin, fmax) = computeMinAndMaxValues(
        pixel_array, image_width, image_height)

    a = 1 if fmin == fmax else 255/(fmax - fmin)
    b = 0 - fmin * a

    res = createInitializedGreyscalePixelArray(image_width, image_height)
    for r in range(image_height):
        for c in range(image_width):
            res[r][c] = round(a*pixel_array[r][c] + b)

    return res


def computeHistogramArbitraryNrBins(pixel_array, image_width, image_height, nr_bins):
    res = [0.0] * nr_bins
    step_size = 256 / nr_bins

    for r in range(image_height):
        for c in range(image_width):
            res[int(pixel_array[r][c] / step_size)] += 1.0

    return res
