# **DogFind**
Welcome! This repo contains my learning process and current knowledge upon the CNN topics that I will be diving into today.<br>
Themes include:
* Data Preprocessing for Image Recognition
* Splitting of your data into training and testing groups
* Creating CNN Model
* Optimisation functions / Loss functions
* Graphing the performance of the model
* And finally, testing the results

## Data Preparation

Let's start from the beginning, since every CNN Model needs to be trained on some sample of data, there comes up a question on where to find it? <br> <br>
Easiest way is usually to use our best friend (Google). My [data](https://www.kaggle.com/competitions/dog-breed-identification/data) is created by Kaggle for one of their past competition, which has over 120+ dog breeds and more than 20,000+ images for both training and testing.
<br><br>
After we download all of the data, now we need to download all of the packages and libraries we will use (there is a lot..) <br>
* Tensorflow
* NumPy
* OpenCV
* Pandas
* Scikit-learn
* Matplotlib
<details>

<summary>GPU Accelaration</summary>
<br>
Additionally if you want to use GPU acceleration, you may need to install Tensorflow CUDA for NVIDIA GPU's or Tensorflow Metal for Apple M series.
Check out the respective documentations for the package you want.
</details>

My recommendation would be to download Conda (or Miniconda for more tech-savvy guys) and use its commands to download the packages, since it simplifies the stuff for you and there's no need to worry about compatibility issues and missing dependencies.
<br><br>
After we did all of the boring stuff, we can finally start coding!

## Data Preprocessing

Now let's dive into the code and I will be explaining each part with more details.

```
img_size = (331, 331)
img_full_size = (331,331,3)

trainimagefolder = "/Users/yaremayaremcuk/Documents/dogfind/train"
```
After we imported all of the libraries, we specify a couple of parameters that we gonna use in the future.<br><br>
Our model will be trained on the images of size 331x331, which is the size we will be changing all of our data to. <br>

You probably are confused about this line
```
img_full_size = (331,331,3)
```
Where did the 3 comes from? <br><br>
Even though the images will be 331 by 331 pixels, our model would need to see the actual pixels value in order to distinguish the color of the pixel, specifically its RGB channel value.<br>
This way every pixel from our image will be deconstructed into its 3 channels (red, green, blue) and our model will be trained on their values.
<br><br>
Next step is to load our csv table, where our dog breed labels and id's of the corresponding images are constructed in 2 collumns.
```
df = pd.read_csv("/Users/yaremayaremcuk/Documents/dogfind/labels.csv")

print("Head of lables:")
print("===============")
print(df.head())
print(df.describe())

print("Group by labels:")
grouplables = df.groupby("breed")["id"].count()
print(grouplables.head(10))
```
The only meaningful line is df table initilisation, since other lines are just used for showing the data and checking that everything is ok.
<br><br><br>
<img width="438" alt="Screenshot 2024-06-26 at 6 26 11 PM" src="https://github.com/YaremaYaremchuk/DogFind/assets/89141796/d0a1ef90-7700-43e3-932b-e35c0367a575">

As we can see we have 10,222 images id's and 120 unique breeds, which hints that we loaded the right dataset🥳
<br><br>
After that we quickly check if we can load the image through opencv package and initialise our arrays for images and labels.<br>
```
imgpath = "/Users/yaremayaremcuk/Documents/dogfind/train/0a0c223352985ec154fd604d7ddceabd.jpg"
img = cv2.imread(imgpath)

allimg = []
alllabels = []

```
If no errors popped up, we are ready to move to the main process.
<br>
```
for idx , (image_name, breed) in enumerate(df[['id' , 'breed']].values):
    img_path = os.path.join(trainimagefolder, image_name + '.jpg')
    print(img_path)


    img = cv2.imread(img_path)
    resized = cv2.resize(img, img_size, interpolation = cv2.INTER_AREA)
    allimg.append(resized)
    alllabels.append(breed)
```
It looks scary at first, but what this loop actually does is letting us iterate through the dataframe and get the image id and breed name for every collumn in it.
<br>
Every image location is "it's id" + ".jpg" extension. <br><br>
After all of the resizings we append the image to our array, as well as its label to the label array.<br><br>
To answer some questions regarding what is actually we are adding to the arrays and how that data looks, we can easily print it out for learning purposes.<br><br>
<img width="146" alt="Screenshot 2024-06-26 at 6 43 32 PM" src="https://github.com/YaremaYaremchuk/DogFind/assets/89141796/cd34bba2-c505-45b5-82b8-72bf6b22d5ce"><br>
Our "resized" image is basically a 3 dimensional array of (331,331,3), where every value is between 0-255 (RGB range), and our "breed" is just the breed name.<br><br>
Hope it cleared out any confusion you may have had🤔.
<br><br>
You may breathe out, as everything left is to simply save the final arrays into NumPy files in order to use them later on.
```
np.save("/Users/yaremayaremcuk/Documents/data/allimages.npy",allimg)
np.save("/Users/yaremayaremcuk/Documents/data/alllabels.npy",alllabels)
```

## Model Creation and Training

Let's dive into the main part that we all been waiting for, **CNN Model**
<br><br>
*If you plan on using GPU for training, you may include these lines to be sure that Tensorflow sees your GPU*
```
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0],True)
```
<br>We start with initialising all of our parameters and loading our previously created image and label arrays, we normalise them to the [0-1] range by diving each value by 255.0, in order to simplify and speed up the training process.<br>

```
img_size = (331, 331)
img_full_size = (331,331,3)
batchsize = 8

allimages = np.load("/Users/yaremayaremcuk/Documents/data/allimages.npy")
alllabels = np.load("/Users/yaremayaremcuk/Documents/data/alllabels.npy")

allimagesmod = allimages / 255.0
```
<br>
Next couple lines are a bit confusing, but stay with me.<br><br>

```
le = LabelEncoder()
integerlables = le.fit_transform(alllabels)

numofcateg = len(list(le.classes_))

alllabelsmod = to_categorical(integerlables, num_classes=numofcateg)
```

Here we are utilizing scikit-learn library to transform all of our breed names into a type of dictionary that maps (through hash functions) the names to numerical values.<br>
The length of it is 120, which is the number of unique categories we have.
<br><br>
Some of you may ask, why do we even need to do all of this label manipulation? <br>
Machine Learning essence is basically curve fitting all of our data points (our images) and, mathematically, we can't do any computations to the string 'pug' or 'pitbull'.<br>
This is why we need a this sort of function to transform the string values to the numbers, which then are transformed into binary matrix through "to_categorical" function.
<br><br>
*For curious one, binary matrix looks like this*
```
a = keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
print(a)
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
```
<br>
Even more curios one may ask, why do we need binary matrix and why we can't just use our integerlables array?<br><br>
Answer is in how categorical loss functions are calculated.
<br><br>
<img width="754" alt="Screenshot 2024-06-26 at 7 50 22 PM" src="https://github.com/YaremaYaremchuk/DogFind/assets/89141796/d713c37f-8a8c-4281-ae8a-66b781f3af11">
<img width="751" alt="Screenshot 2024-06-26 at 7 50 55 PM" src="https://github.com/YaremaYaremchuk/DogFind/assets/89141796/a2997a1f-f9dc-4b7b-8a7e-b47a779f909b">

Here are a couple images I found on internet that may help to visualise the process.<br><br>
Basically our model produces an output vector with the length of number of our classes/categories.<br>
This array is generated through softmax activation function in a way that all values in it adds up to 1, think of it of a probability-wise if you want<br><br>
Those values are then taken to calculate our loss difference from the actual value (which is a binary matrix of the labels). <br> Ideally, our model output vector will be the same as label's one, but we know no one is perfect (even our model trained on thousands of images😅).
<br><br>
Now let's jump into preparing our training data!
```
X_train, X_test, y_train, y_test = train_test_split(allimagesmod, alllabelsmod, test_size=0.3, random_state=42)
```
Here we simply split all of our images and labels, based on the parameters, into two parts: train and test.<br><br>
This may take a while, and it is where I got my first issues with the data, since I couldn't load all of the images and labels arrays into the RAM and I was running out of memory.<br>
If you think about it, each image is 328,683 pixel values(331*331*3) and we have 10,223 of these images.... <br><br>
So after couple hours of testing, I got the maximum number of images I could load on my Mac was 8,500, which is still large enough to train the model.<br><br>

After all of this hard work on our memory, we can finally free it up by deleting the old data.
```
del allimages
del alllabels
del integerlables
del allimagesmod
del alllabelsmod
```
Now we load our CNN Model called NASNetLarge, which is one of the best pre-trained models in Image Classification.<br>
```
mymodel = NASNetLarge(input_shape=img_full_size , weights='imagenet', include_top=False)

for layer in mymodel.layers:
    layer.trainable = False
    print(layer.name)
```
We also disable the training for all of the existing layers (what's the reason of using pre-trained model then😁)<br><br>























