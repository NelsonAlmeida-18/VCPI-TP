import numpy as np
from PIL import Image, ImageEnhance
import cv2

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
        img = img.transform(img.size, Image.AFFINE, skewMatrix)

    else:
        img = cv2.warpAffine(img, skewMatrix, (self.imageWidth-5, self.imageHeight-5))

    return img

def rotateImage(self,img):
    # Lets rotate the image
    # Lets generate a random angle
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

def addBluring(self,img):
    # In this transformation we will add bluring to the image
    # Lets generate a random bluring factor
    # We will use this to simulate motion blur effects / out of focus images
    bluringFactor = np.random.randint(1, 5)
    img = cv2.GaussianBlur(img, (bluringFactor, bluringFactor), 0)

    return img



def addGaussianNoise(self,img):
    pass
    

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
