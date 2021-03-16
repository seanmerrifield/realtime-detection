import numpy as np
import cv2
'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    #image = image.reshape(1, 3, height, width)
    image = image.reshape(1, *image.shape)
    return image


def get_mask(image_shape, box_coords):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(image_shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask


def create_output_image(image, output, color='blue', threshold=0.5):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''

    color = color.lower()
    color_channels = {"blue": (255,0,0), "green": (0,255,0), "red": (0,0,255)}
    if color not in color_channels: raise Exception(f"Color: {color} is not a valid color option (must be blue, green, or red)")

    #Remove excess dimensions so that it is a list of bounding boxes
    output = output.squeeze()

    for image_id, label, conf, x_min, y_min, x_max, y_max in output:

        # Skip if confidence level is below 0.5
        if conf < threshold: continue

        x_min = int(x_min * image.shape[1])
        x_max = int(x_max * image.shape[1])
        y_min = int(y_min * image.shape[0])
        y_max = int(y_max * image.shape[0])
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color_channels[color], 2) 


    return image
    