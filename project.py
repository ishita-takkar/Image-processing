"""
DSC 20 Project
Name(s): Ishita Takkar, Aarshia Gupta
PID(s):  A17859782, A17920554
Sources: lectures and discussions
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        if not (isinstance(pixels, list) and len(pixels)>=1):
            raise TypeError()
        if not all([isinstance(row, list) and len(row)>=1 for row in pixels]):
            raise TypeError()
        if not all([len(row) == len(pixels[0]) for row in pixels]):
            raise TypeError()
        if not all([isinstance(pixel, list) and len(pixel)==3 for row in \
    pixels for pixel in row]):
            raise TypeError()
        max_intensity = 255
        if not all([0<=i<=max_intensity for row in pixels for pixel in row \
    for i in pixel]):
            raise ValueError()

        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[i for i in pixel]for pixel in row]for row in self.pixels]
    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        copy_pixels = self.get_pixels()
        return RGBImage(copy_pixels) 

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if not isinstance(row, int):
            raise TypeError()
        if not isinstance(col, int):
            raise TypeError()
        if not (0 <= row < self.num_rows) or not (0 <= col< self.num_cols):
            raise ValueError()
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError
       
        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if not (0 <= row < self.num_rows) or not (0 <= col < self.num_cols):
            raise ValueError()
        new_length = 3
        if not isinstance(new_color, tuple) or not len(new_color)==new_length \
    or not all((isinstance(elem, int) for elem in new_color)):
            raise TypeError()
        valid_intensity = 255
        if not all( i<=valid_intensity for i in new_color):
            raise ValueError()
        
        for i in range(new_length):
            if new_color[i]>=0:
                self.pixels[row][col][i]= new_color[i]


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0
    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                           # 1
        >>> img = img_read_helper('img/test_image_32x32.png')               # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')# 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        negation_val = 255
        neg_pixels = [[[negation_val - val for val in pixel]for pixel in row]\
    for row in image.get_pixels()]
        negate_img = RGBImage(neg_pixels)
        return negate_img
    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        avg_pixel = 3
        gray_pixels = [[[sum(pixel)//avg_pixel]*avg_pixel for pixel in row]\
    for row in image.get_pixels()]
        grayscale_img = RGBImage(gray_pixels)
        return grayscale_img

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        rotating = image.get_pixels()[::-1]
        rotate_pixel = [row[::-1] for row in rotating]
        return RGBImage(rotate_pixel)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        sum_pixels = [[sum(pixel)//len(pixel) for pixel in row]\
    for row in image.get_pixels()]
        avg_brightness = sum(sum(pixel) for pixel in \
    sum_pixels)//sum(len(pixel) for pixel in sum_pixels)
        return avg_brightness

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        if not isinstance(intensity, int):
            raise TypeError()
        max_intensity= 255
        min_intensity = -255
        if not (min_intensity <= intensity <= max_intensity):
            raise ValueError()
        adjust_pixels = [[[max(0, min(val+intensity,255))for val in pixel]\
    for pixel in row] for row in image.get_pixels()]
        adjust_img = RGBImage(adjust_pixels)
        return adjust_img

    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        blur_pixels = []

        for row in range(image.num_rows):
            blur_row = []
            for col in range(image.num_cols):
                r_sum, g_sum, b_sum = 0, 0, 0
                
                neighbours = 2
                neighbouring = [(x,y) for x in range(max(0, row - 1), \
min(image.num_rows, row + neighbours)) for y in range(max(0, col - 1),\
min(image.num_cols, col + neighbours))]
                for x,y in neighbouring:
                    r, g, b = image.pixels[x][y]
                    r_sum += r
                    g_sum += g
                    b_sum += b

                r_avg = r_sum // len(neighbouring)
                g_avg = g_sum // len(neighbouring)
                b_avg = b_sum // len(neighbouring)

                blur_row.append([r_avg, g_avg, b_avg])

            blur_pixels.append(blur_row)

        return RGBImage(blur_pixels)


#Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.freecall=0
        self.cost = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        self.cost+=5
        if self.freecall >0:
            self.freecall-=1
            self.cost-=5
            return super().negate(image)
        else:
            return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        self.cost+=6
        if self.freecall>0:
            self.freecall-=1
            self.cost-=6
            return super().grayscale(image)
        else:
            return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        self.cost+=10
        if self.freecall>0:
            self.freecall-=1
            self.cost-=10
            return super().rotate_180(image)
        else:
            return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        self.cost+=1
        if self.freecall>0:
            self.freecall-=1
            self.cost-=1
            return super().adjust_brightness(image, intensity)
        else:
            return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        self.cost+=5
        if self.freecall>0:
            self.freecall-=1
            self.cost-=5
            return super().blur(image)
        else:
            return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if not isinstance(amount, int):
            raise TypeError()
        if amount<=0:
            raise ValueError()

        self.freecall+=amount



# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        super().__init__() #do we need to call it
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)
        """
        if not isinstance(chroma_image, RGBImage) or not \
isinstance(background_image, RGBImage):
            raise TypeError()

        if chroma_image.size() != background_image.size():
            raise ValueError()

        new_image = []

        for i in range(chroma_image.num_rows):
            row = []
            for j in range(chroma_image.num_cols):
                chroma_img_pixels = chroma_image.get_pixel(i, j)
                bg_img_pixels = background_image.get_pixel(i, j)
                if chroma_img_pixels == color:
                    row.append(list(bg_img_pixels))
                else:
                    row.append(list(chroma_img_pixels)) 
            new_image.append(row)

        return RGBImage(new_image)


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        if not isinstance(sticker_image, RGBImage) or not \
isinstance(background_image, RGBImage):
            raise TypeError()

        if background_image.num_cols < sticker_image.num_cols or \
        background_image.num_rows < sticker_image.num_rows:
            raise ValueError()

        if not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError()

        if x_pos < 0 or y_pos < 0:
            raise ValueError()

        if background_image.num_rows < y_pos + sticker_image.num_rows or \
        background_image.num_cols < x_pos + sticker_image.num_cols:
            raise ValueError()

        bg_copy = background_image.copy()

        for i in range(sticker_image.num_rows):
            for j in range(sticker_image.num_cols):
                sticker_pixels = sticker_image.get_pixel(i, j)

                bg_copy.set_pixel(y_pos+i, x_pos+j, sticker_pixels)

        return bg_copy  
        

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        copy_img = image.copy()
        single_val = [[sum(pixel)//3 for pixel in row]for row in copy_img.get_pixels()]
        kernel = [
            [-1,-1,-1],
            [-1, 8, -1],
            [-1,-1,-1]
        ]
        image = []
        for row in range(len(single_val)):
            edge = []
            for col in range(len(single_val[0])):
                edgeval = 0
                for x in range(-1,2):
                    for y in range(-1,2):
                        if 0<= row+x< len(single_val) and 0<= col+y< len(single_val[0]):
                            edgeval+= single_val[row+x][col+y]*kernel[x+1][y+1]
                if edgeval<0:
                    edgeval=0
                elif edgeval>255:
                    edgeval = 255
                edge.append(edgeval)
            image.append(edge)

        output = [[[value, value, value]for value in row]for row in image]
        return RGBImage(output)      

# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        self.k_neighbors = k_neighbors
        self.data = []
    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        if len(data)< self.k_neighbors:
            raise ValueError()
        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError()
        if not image1.size()==image2.size():
            raise ValueError()
        dist = sum([(image1.get_pixel(row, col)[pixel] - image2.get_pixel(row, col)[pixel])**2 for row in range(image1.num_rows) for col in range(image1.num_cols) for pixel in range(0,3)])
        return dist**0.5

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        most_label = {i:candidates.count(i) for i in candidates}
        maxi_val= max(most_label.values())
        for key, val in most_label.items():
            if maxi_val== val:
                return key

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        if self.data == []:
            raise ValueError()
        data_dist = sorted([(self.distance(image,images), labels) for images, labels in self.data])[0:self.k_neighbors]
        label = self.vote([label for dist, label in data_dist])
        return label



def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
