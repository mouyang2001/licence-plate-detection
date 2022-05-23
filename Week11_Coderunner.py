def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(1, image_height-1):
        for c in range(1, image_width-1):
            num = (pixel_array[r-1][c+1] + 2*pixel_array[r][c+1] + pixel_array[r+1][c+1]
                   - pixel_array[r-1][c-1] - 2*pixel_array[r][c-1] - pixel_array[r+1][c-1])
            image[r][c] = abs(num/8)

    return image


def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(1, image_height-1):
        for c in range(1, image_width-1):
            num = (pixel_array[r-1][c-1] + 2*pixel_array[r-1][c] + pixel_array[r-1][c+1]
                   - pixel_array[r+1][c-1] - 2*pixel_array[r+1][c] - pixel_array[r+1][c+1])
            image[r][c] = abs(num / 8)

    return image


def computeBoxAveraging3x3(pixel_array, image_width, image_height):
    mean_array = createInitializedGreyscalePixelArray(
        image_width, image_height)

    for r in range(1, image_height-1):
        for c in range(1, image_width-1):
            total = 0.0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    total += pixel_array[r+i][c+j]

            mean_array[r][c] = total / 9

    return mean_array


def computeMedian5x3ZeroPadding(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(image_height):
        for c in range(image_width):
            values = []
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    if (r+i < 0 or r+i >= image_height or c+j < 0 or c+j >= image_width):
                        values.append(0)
                    else:
                        values.append(pixel_array[r+i][c+j])

            values.sort()
            index = (len(values)-1) // 2
            image[r][c] = values[index]

    return image


def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)
    kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1]

    for r in range(image_height):
        for c in range(image_width):
            values = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    m, n = r+i, c+j

                    if m < 0:
                        m = 0
                    if n < 0:
                        n = 0
                    if m >= image_height:
                        m = image_height-1
                    if n >= image_width:
                        n = image_width-1

                    values.append(pixel_array[m][n])

            total = 0.0
            for x in range(9):
                total += values[x] * kernel[x]

            image[r][c] = round(total/16, 2)

    return image


def computeStandardDeviationImage3x3(pixel_array, image_width, image_height):
    image = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(1, image_height-1):
        for c in range(1, image_width-1):
            nums = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    nums.append(pixel_array[r+i][c+j])

            image[r][c] = get_standard_deviation(nums)

    return image


def get_standard_deviation(nums):
    length = len(nums)
    mean = sum(nums) / length
    variance = sum((x - mean)**2 for x in nums) / length
    standard_deviation = variance ** 0.5
    return standard_deviation
