import sys
import numpy as np
from skimage import io


def reconstruction(image_input, average, eig_vec):
    #image is the raw photo without processing
    #reconstruct image use 1st~4nd eigenface
    
    img = np.asarray(image_input)
    img = img.flatten()
    
    output = average
    for i in range(4):
        weight = np.dot((img - average), np.transpose(eig_vec[:, i]))
        output = output + weight * eig_vec[:, i]
        
    output = (output - average)
    output = output + average 
    output -= np.min(output)
    output /= np.max(output)
    output = (output * 255).astype(np.uint8)
    output = np.reshape(output, (600, 600, 3))

    return output

img = []
for i in range(415):
    img.append(io.imread(sys.argv[1] + '/' + str(i) + '.jpg'))

image = np.asarray(img)
image = np.reshape(image, (415, -1))

average_image = np.mean(image, axis = 0)

var = image - average_image

alt_cov = np.dot(var, np.transpose(var))

eig_val, eig_vec = np.linalg.eig(alt_cov)

eig_vec = np.dot(np.transpose(var), eig_vec).astype(np.float64)
eig_vec = eig_vec / np.linalg.norm(eig_vec, axis = 0)

sequence = reconstruction(io.imread(sys.argv[1] + '/' + sys.argv[2]), average_image, eig_vec)

io.imsave('reconstruction.jpg', sequence)
