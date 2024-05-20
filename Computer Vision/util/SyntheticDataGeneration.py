import  cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm


class Transformations():
    def __init__(self):
        pass

    def transformImage(self, img, imgWidth, imageHeight):
        self.imageWidth = imgWidth  
        self.imageHeight = imageHeight
        
        # lets initialize all the possible transformations
        transformations = {
            "skew": self.skewImage,
            "rotate": self.rotateImage,
            "brightness": self.changeBrightness,
            "perlin": self.addPerlinNoise,
            "gaussian": self.addGaussianNoise,
            "blur": self.addGaussianBlur,
            "colorspace": self.colorspaceTransform
        }

        # Lets get a random number of transformations to apply to the image
        nTransformations = np.random.randint(1, 7)
        # Lets get nTransformation random transformations to apply
        transformationsToApply = np.random.choice(list(transformations.keys()), nTransformations, replace=False)
        
        img = np.array(img)

        for transformation in transformationsToApply:
            img = transformations[transformation](img)

        return img

    def getImageBrightness(self,img):
        # Lets calculate the brightness of the image
        brightness = np.mean(img)

        return brightness
    
    def skewImage(self,img):
        # Lets skew the image
        # Lets generate a random skew factor
        skewFactor = np.random.uniform(-0.5, 0.5)

        # Lets skew the image
        skewMatrix = np.float32([[1, skewFactor, 0], [0, 1, 0]])
        if isinstance(img, Image.Image):
            # Lets convert the image to a numpy array
            img = np.array(img)
        
            img = cv2.warpAffine(img, skewMatrix, (self.imageWidth-5, self.imageHeight-5))

        return img
    
    def rotateImage(self,img):
        # Lets rotate the image
        # Lets generate a random angle
        if isinstance(img, Image.Image):
            angle = np.random.randint(0, 45)
            img = img.rotate(angle)
        else: 
            angle = np.random.randint(0, 45)

            # Lets rotate the image
            rotationMatrix = cv2.getRotationMatrix2D((self.imageWidth / 2, self.imageHeight / 2), angle, 1)
            img = cv2.warpAffine(img, rotationMatrix, (self.imageWidth, self.imageHeight))

        return img
    

    def changeBrightness(self,img):
        if isinstance(img, Image.Image):
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(np.random.uniform(0.5, 1.5))

        return img
    
    def colorspaceTransform(self,img):
        # In this transformation we will change the color space of the image
        # We will convert the image to HSL and change the saturation of the image
        # This aims to simulate the effect of different lighting conditions
        # Lets generate a random saturation factor
        saturation = np.random.uniform(0.5, 1.5)
        if isinstance(img, Image.Image):
            img = img.convert("HSV")
            img = img.point(lambda i: i * saturation)
            img = img.convert("RGB")
        
        else:
        
            # Lets convert the image to HSL
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

            # Lets change the saturation of the image
            img[:,:,2] = img[:,:,2] * saturation

            # Lets convert the image back to BGR
            img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)

        return img
    
    def noiseTransform(self,img):
        # In this transformation we will add diverse types of noise to the image
        # For noise types we will use perlin and gaussian noise to simulate imperfections in the image
        # Lets generate a random noise type
        noiseType = np.random.choice(["perlin", "gaussian"])

        if noiseType == "perlin":
            img = self.addPerlinNoise(img)
        else:
            img = self.addGaussianNoise(img)

        return img
    
    def addGaussianBlur(self,img):
        # In this transformation we will add bluring to the image
        # Lets generate a random bluring factor
        # We will use this to simulate motion blur effects / out of focus images
        if isinstance(img, Image.Image):
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.randint(1, 5)))
        else:
            bluringFactor = np.random.randint(1, 5)
            # img = cv2.GaussianBlur(img, (bluringFactor, bluringFactor), 0)

        return img
    

    
    def addGaussianNoise(self,img):
        # In this transformation we will add gaussian noise to the image
        # Lets generate a random noise factor
        if isinstance(img, Image.Image):
            noiseFactor = np.random.uniform(0, 100)
            noise = np.random.normal(0, noiseFactor, (self.imageHeight, self.imageWidth, 3))
            noise = noise.astype(np.uint8)
            noise = Image.fromarray(noise)
            # img = Image.blend(img, noise, 0.5)


        return img
        
    
    # TODO: Add a stochastic parameter to the perlin noise function in order to make the noise diferent
    def addPerlinNoise(self, image):

        # Lets verify if the image is a PIL image instance
        if not isinstance(image, Image.Image):
            # Lets normalize the signal array
            image = image - np.min(image)
            image = image / np.max(image)
            image = (image * 255).astype(np.uint8)
    
            image = Image.fromarray(image)

        perlinNoise = self.perlin()
        # Lets overlay the perlin noise to the image but alter the transparency of the perlin
        perlinNoise = perlinNoise * 255
        perlinImage = Image.fromarray(perlinNoise)

        # Lets convert the image to black and white
        perlinImage = perlinImage.convert("RGBA")

        perlinImage.resize(image.size)
        
        # Lets get the alpha channel of the perlin noise image
        alpha = perlinImage.getbands().index('R')
        alpha = perlinImage.split()[alpha]


        image.paste(perlinImage, (0, 0), alpha)

        return image


    def perlin(self):
        np.random.seed(0)
        # Min 5x5, max 20x20, res has to be a multiple of the shape
        return self.generate_fractal_noise_2d((480, 480), (20, 20))
        

    # From https://github.com/pvigier/perlin-numpy
    def interpolant(self,t):
        return t*t*t*(t*(t*6 - 15) + 10)


    def generate_perlin_noise_2d(self,
            shape, res, tileable=(False, False)
    ):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
                .transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        if tileable[0]:
            gradients[-1,:] = gradients[0,:]
        if tileable[1]:
            gradients[:,-1] = gradients[:,0]
        gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
        g00 = gradients[    :-d[0],    :-d[1]]
        g10 = gradients[d[0]:     ,    :-d[1]]
        g01 = gradients[    :-d[0],d[1]:     ]
        g11 = gradients[d[0]:     ,d[1]:     ]
        # Ramps
        n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = self.interpolant(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


    def generate_fractal_noise_2d(self,shape, res, octaves=1, persistence=0.5, lacunarity=2, tileable=(False, False),
    ):
        noise = np.zeros(shape)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.generate_perlin_noise_2d(
                shape, (frequency*res[0], frequency*res[1]), tileable
            )
            frequency *= lacunarity
            amplitude *= persistence
        return noise


class SyntheticDataGeneration():
    def __init__(self, templateImagesPath="../data/German_Templates/", syntheticDataPath="../data/SyntheticData/", trainingImagesPath = "../data/train/Final_Training/Images/"):
        # Lets prepare the ambient to receive the images
        self.templateImagesPath = templateImagesPath
        self.syntheticDataPath = syntheticDataPath
        self.trainingImagesPath = trainingImagesPath

        if not os.path.exists(self.syntheticDataPath):
            os.makedirs(self.syntheticDataPath)

        self.imageTransformer = Transformations()
        
        self.generate()
        # self.testTransformations()

    def testTransformations(self):
        # Lets test the transformations
        img = Image.open("../data/German_Templates/0.png")
        imgWidth, imgHeight = img.size

        img = self.imageTransformer.transformImage(img, imgWidth, imgHeight)
        syntheticBG = self.generateSyntheticBackground((imgWidth, imgHeight))
        img = self.merge(syntheticBG, img)

        plt.imshow(img)
        plt.show()

    def imagesPerSignal(self):
        
        trainingImagesPath = self.trainingImagesPath
        trainingImagesPaths = os.listdir(trainingImagesPath)
        
        #Lets get the number of signals to generate
        # For this we will consult the number of images in the training dataset for each signal
        # Then we will get the max number of images and generate the same number of images for each signal + 200 images
        # This is to ensure that we have enough images for each signal
        maxImages = 0
        imagesPerSignal = {}
        for signal in trainingImagesPaths:
            
            if not os.path.isdir(f"{trainingImagesPath}{signal}"):
                continue

            images = os.listdir(f"{trainingImagesPath}{signal}")
            imagesPerSignal[signal] = len(images)
            
            if len(images) > maxImages:
                maxImages = len(images)
        
        for signal in imagesPerSignal.keys():
            imagesPerSignal[signal] = max(0, abs(imagesPerSignal[signal]-maxImages)) + 200

        return imagesPerSignal


        
    def generate(self):
        self.ImagePath = self.templateImagesPath
        signals = os.listdir(self.ImagePath)

        imagesPerSignal = self.imagesPerSignal()

        for signal in signals:
            signalName = signal.split(".")[0]
            signalName = signalName.split("_")[0]

            padding = 5 - len(signalName)
            signalName = "0"*padding + signalName

            
            # Create a directory to store the images
            if not os.path.exists(f"{self.syntheticDataPath}/{signalName}"):
                os.makedirs(f"{self.syntheticDataPath}/{signalName}")
            
            if os.path.exists(f"{self.syntheticDataPath}/{signalName}") and len(os.listdir(f"{self.syntheticDataPath}/{signalName}")) > 0:
                print(f"Synthetic data for signal {signal} already generated, skipping...")
                continue
            
            print(f"Generating {imagesPerSignal[signalName]} images for signal {signal}...")

            for i in tqdm(range(imagesPerSignal[signalName])):
                signalImage = Image.open(f"{self.ImagePath}{signal}")
                self.imageWidth, self.imageHeight = signalImage.size

                signalImage = self.imageTransformer.transformImage(signalImage, self.imageWidth, self.imageHeight)

                syntheticBg = self.generateSyntheticBackground((self.imageWidth-1, self.imageHeight-1))
                mergedImage = self.merge(syntheticBg, signalImage)

                mergedImage = mergedImage.resize((28, 28))

                mergedImage.save(f"{self.syntheticDataPath}/{signalName}/{signalName}_{i}.ppm")

        print("Synthetic data generation finished!")
    
    # TODO: fix the transformSignal function, the image is not being loaded correctly
    def transformSignal(self, signal):

        # Lets resize the signal image
        shrinkedSize = np.random.randint(low=signal.size[0]/2, high=signal.size[0])
        print("Shrinked size: ", shrinkedSize)
        signal = signal.resize(
            (shrinkedSize, shrinkedSize)
        )

        return signal

        
    def merge(self, syntheticBG, signal):
        width, height = syntheticBG.size
        mergedImage = Image.new("RGBA", syntheticBG.size)

        # Lets convert the signal to a PIL image
        if not isinstance(signal, Image.Image):
            # Lets normalize the signal array
            signal = signal - np.min(signal)
            signal = signal / np.max(signal)
            signal = (signal * 255).astype(np.uint8)
    
            signal = Image.fromarray(signal)

        signal = signal.convert("RGBA")

        alpha = signal.getbands().index('A')
        alpha = signal.split()[alpha]
        
        mergedImage.paste(syntheticBG, (0, 0))
        
        x = np.random.randint(0, abs(width - signal.size[0])+1)
        y = np.random.randint(0, abs(height - signal.size[1])+1)
        # Calculate the maximum allowed offset for the signal
        max_offset_x = int(0.15 * signal.size[0])
        max_offset_y = int(0.15 * signal.size[1])

        # Generate random offsets within the allowed range
        offset_x = np.random.randint(-max_offset_x, max_offset_x)
        offset_y = np.random.randint(-max_offset_y, max_offset_y)

        # Calculate the final position of the signal
        x = max(0, min(x + offset_x, width - signal.size[0]))
        y = max(0, min(y + offset_y, height - signal.size[1]))

        # Paste the signal into the background with the mask
        mergedImage.paste(signal, (x, y), alpha)

        # Resize the result to a 28x28 image
        # mergedImage = mergedImage.resize((28, 28))

        enhancer = ImageEnhance.Sharpness(mergedImage)
        enhancedImage = enhancer.enhance(3.0)

        return mergedImage

    def generateSyntheticBackground(self,image_size=(28, 28)):

        # Lets add a random background color to the image
        bgColor = np.random.randint(0, 255, 3)

        # Lets use a blank canvas to add the background color
        background = Image.new(mode='RGB', size=image_size, color = (bgColor[0], bgColor[1], bgColor[2]))

        return background
