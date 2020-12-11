# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as imag

imgClasses = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse","Ship", "Truck"]



# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(32, 32))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def load_from_folder(folderName):
    
    for x in range(20):
        print('Image :' + str(x+1))
        # load the image
        img = load_image(folderName+'/' + str(x+1) +'.jpg')
        # load model
        model = load_model('final_model.h5')
        # predict the class
        path = '32x32_Img/' + str(x+1) +'.jpg'
        result = model.predict_classes(img)
        plt.imshow(imag.imread(folderName+'/' + str(x+1) +'.jpg'))
        plt.show()
        plt.imshow(imag.imread('32x32_Img/' + str(x+1) +'.jpg'))
        plt.show()
        print("Image Classified as : ", end='')
        rst = result[0]
        print(imgClasses[rst])

# entry point, run the example
load_from_folder('Testing')
