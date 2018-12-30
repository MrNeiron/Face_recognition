from os import listdir
import numpy as np
import cv2

def preprocess_image2(input_image_path, output_image_path, model_image_size, grayscale = False, save = False, resize = True):
     
    image = cv2.imread(input_image_path, 0 if grayscale else 1)
    if resize:
        resized_image = cv2.resize(image, model_image_size)#if resize else image
    else:
        resized_image = image
    
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.

    if grayscale: resized_image = np.expand_dims(resized_image, 3)# Add last axis
    image_data = np.expand_dims(resized_image, 0)  # Add batch dimension.
        
    if (save):
        cv2.imwrite(output_image_path, resized_image)
        
    return image_data
    
    
def take_n_resize_images(input_path, output_path = "", image_size=(28,28), num_examples=None, grayscale = False, save = False, resize = True, start = 0):

    if num_examples == None: num_examples = len(listdir(input_path))
    images = np.zeros((num_examples,
                       image_size[0],
                       image_size[1],
                       1 if grayscale else 3))

    for i,file in enumerate(listdir(input_path)):
        #print("filename = ", file)
        if i < start: continue
        if i == start+num_examples: break
        images[i-start] = preprocess_image2(input_path +'/'+file, 
                                     output_path +'/'+file, 
                                     image_size,
                                     grayscale,
                                     save,
                                     resize)


    return images

if __name__ == "__main__":
    imgs = take_n_resize_images("datasets/myImages",
                         image_size = (96,96),
                         num_examples = 3,
                         start = 20,
                         grayscale = True,
                         save = False)
    print(imgs.shape)