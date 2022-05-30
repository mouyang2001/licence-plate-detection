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

            erosion[r][c] = pixel_value

    return erosion


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    dilation = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(image_height):
        for c in range(image_width):

            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if (r+i in range(image_height) 
                    and c+j in range(image_width) 
                    and pixel_array[r+i][c+j] != 0):
                        dilation[r][c] = 1

    return dilation


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    dictionary = {}

    visited = set()
    directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]

    label = 1
    for r in range(image_height):
        for c in range(image_width):
            if pixel_array[r][c] == 0 or (r, c) in visited:
                continue

            queue = Queue()
            queue.enqueue((r, c))
            visited.add((r, c))

            while not queue.isEmpty():
                (row, col) = queue.dequeue()
                dictionary[label] = dictionary.get(label, 0) + 1
                result[row][col] = label

                for (dr, dc) in directions:
                    r1, c1 = row + dr, col + dc
                    if (r1 in range(image_height) and c1 in range(image_width)
                            and pixel_array[r1][c1] != 0 and (r1, c1) not in visited):
                        queue.enqueue((r1, c1))
                        visited.add((r1, c1))

            label += 1

    return (result, dictionary)
