##################
## Ulku Meteriz ##
##################
from PIL import Image
import numpy as np

# Load image into numpy array
def load_image( filename ) :
    img = Image.open( filename )
    img = img.convert('L')
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

# Scale the data between 0 and 255.
def scale_image(data):
    scaled_data = data - data.min()
    scaled_data = (scaled_data / scaled_data.max()) * 255.0
    scaled_data = np.uint8(scaled_data)
    return scaled_data

def show_image(data, filename=None):
    # Scale the data.
    scaled_data = scale_image(data)
    img = Image.fromarray(scaled_data)
    # If a filename is given, save the image.
    if filename:
        img.save(filename)
    # Show the image.
    img.show()

def save_image(data, filename):
    # Scale the data.
    scaled_data = scale_image(data)
    img = Image.fromarray(scaled_data)
    img.save(filename)

def flip_filter(filter):
    flipped_filter = np.zeros(filter.shape, dtype=filter.dtype)
    # get shape of the filter
    y_shape, x_shape = filter.shape
    
    # flip across the axises
    for y in range(y_shape):
        for x in range(x_shape):
            flipped_filter[y, x] = filter[ (y_shape-1 -y), (x_shape-1 -x) ]
    
    return flipped_filter

# Multiply corresponding elements in the given matrices and return the result.
def elementwise_multiplication(m1, m2):
    assert m1.shape == m2.shape
    result = np.zeros(m1.shape)
    y_shape, x_shape = result.shape
    for y in range(y_shape):
        for x in range(x_shape):
            result[y,x] = m1[y,x] * m2[y,x]
    return result

# Sum up the elements in m.
def sum_up(m):
    result = 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            result += m[i,j]
    return result

def correlation(data, f):
    # Get the image dimensions.
    data_y_shape, data_x_shape = data.shape
    # Get the filter dimensions.
    filter_y_shape, filter_x_shape = f.shape
    # Compute the valid convolution dimension.
    result_shape = (data_y_shape - filter_y_shape + 1 , data_x_shape - filter_x_shape + 1)
    # Initilize the result.
    result = np.zeros(result_shape, dtype=f.dtype)

    y_shape, x_shape = result_shape
    # For every element in resulting matrix...
    for y in range(y_shape):
        for x in range(x_shape):
            # Get the image patch to correlate.
            img_patch = data[y:y + filter_y_shape, x:x + filter_x_shape]
            if img_patch.shape != f.shape:
                continue
            # Perform elementwise multiplication with the filter and sum the results up.
            result[y, x] = sum_up(elementwise_multiplication(img_patch, f))

    return result


# Convolution
def convolve(data, f):
    # flip the filter
    flipped_filter = flip_filter(f)
    # apply correlation
    return correlation(data, flipped_filter)


def edge_response(x_edges, y_edges):
    assert x_edges.shape == y_edges.shape
    response = np.zeros(x_edges.shape, dtype=x_edges.dtype)
    # Get the square of x gradient components.
    x_square = elementwise_multiplication(x_edges, x_edges)
    # Get the square of y gradient components. 
    y_square = elementwise_multiplication(y_edges, y_edges)
    # Sum the squared gradient components.
    response = x_square + y_square
    # Return the squareroot of the response.
    return np.sqrt(response)


def apply_threshold(response, t=100):
    result = np.zeros(response.shape)
    shape_y, shape_x = response.shape
    for y in range(shape_y):
        for x in range(shape_x):
            # If response is bigger than the threshold...
            if response[y, x] > t:
                result[y, x] = 255.0
            # Otherwise ...
            else:  
                result[y, x] = 0.0
    return result

# Sobel edge detector.
def sobel_edge_detection(data, show_images=True):
    x_filter = np.array([ \
        [1.0, 0.0, -1.0], \
        [2.0, 0.0, -2.0], \
        [1.0, 0.0, -1.0]])
    
    y_filter = np.array([ \
        [1.0, 2.0, 1.0], \
        [0.0, 0.0, 0.0], \
        [-1.0, -2.0, -1.0]])

    # Get vertical edges.
    x_edges = convolve(data, x_filter)
    # Get horizontal edges.
    y_edges = convolve(data, y_filter)

    # Compute the edge response.
    response = edge_response(x_edges, y_edges)

    # Apply the threshold, default is 100.
    filtered_response = apply_threshold(response)

    if show_images:
        # Show and save the images.
        show_image(x_edges, filename="x_comp.jpg")
        show_image(y_edges, filename="y_comp.jpg")
        # Show the edge response and the response with threshold.
        show_image(response)
        show_image(filtered_response)

    return x_edges, y_edges, response, filtered_response

def compute_response(k, H):
    # Determinant
    det_H = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    # Trace
    trace_H = H[0][0] + H[1][1]
    return det_H - k * trace_H * trace_H

# Harris corner detector
def harris_corner_detection(data, outfile, W_size=3, k=0.04, threshold=1e11):
    # Compute gradients using sobel edge operator.
    I_x, I_y, _, _ = sobel_edge_detection(data, show_images=False)

    # Compute pixel by pixel products of gradient images.
    I_xx = elementwise_multiplication(I_x,I_x)
    I_yy = elementwise_multiplication(I_y,I_y)
    I_xy = elementwise_multiplication(I_x,I_y)

    # Convolve products with W function full of 1s.
    W = np.ones((W_size,W_size))

    S_xx = convolve(I_xx, W)
    S_yy = convolve(I_yy, W)
    S_xy = convolve(I_xy, W)

    result = np.zeros(S_xx.shape)
    y_shape, x_shape = result.shape
    # For each pixel..
    for y in range(y_shape):   
        for x in range(x_shape):
            # Compute the matrix H.
            H_xy = np.array([ \
                [S_xx[y,x], S_xy[y,x] ], \
                [S_xy[y,x], S_yy[y,x] ]])
            # Compute the response of the detector R
            R = compute_response(k, H_xy)
            result[y,x] = R

    # Apply threshold R.
    result = apply_threshold(result, t=threshold)

    # Save the image.
    save_image(result, outfile)
    return result

################################################################################

## Sobel edge detection
image_data = load_image('../data/image2.jpg')
sobel_edge_detection(image_data)

## Harris corner detection
image_data = load_image('../data/input_hcd1.jpg')
harris_corner_detection(image_data, 'hcd1_output.jpg')

# For experiments on filter size W.
# harris_corner_detection(image_data, 'hcd1_output_5.jpg', W_size=5)
# harris_corner_detection(image_data, 'hcd1_output_7.jpg', W_size=7)

# For experiments on parameter k.
# harris_corner_detection(image_data, 'hcd1_output_05.jpg', k=0.05)
# harris_corner_detection(image_data, 'hcd1_output_5_06.jpg', W_size=5, k=0.06)
# harris_corner_detection(image_data, 'hcd1_output_7_06.jpg', W_size=7, k=0.06)

image_data = load_image('../data/input_hcd2.jpg')
harris_corner_detection(image_data, 'hcd2_output.jpg')

# For experiments on filter size W.
# harris_corner_detection(image_data, 'hcd2_output_5.jpg', W_size=5)
# harris_corner_detection(image_data, 'hcd2_output_7.jpg', W_size=7)

# For experiments on parameter k.
# harris_corner_detection(image_data, 'hcd2_output_06.jpg', k=0.06)
# harris_corner_detection(image_data, 'hcd2_output_5_06.jpg', W_size=5, k=0.06)
# harris_corner_detection(image_data, 'hcd2_output_7_06.jpg', W_size=7, k=0.06)
