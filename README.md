# A Neural Algorithm of Artistic Style

# Introduction
"A Neural Algorithm of Artistic Style" was first published in September 2015 by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. In that paper, they introduced "an artificial system based on Deep Neural Networks that creates artistic images of high perceptual quality."

# Overview
The key finding of their paper is that the representations of the content and style of a given image are separable, suggesting the possibility of mixing different contents and styles from different images to create brand-new images.

The idea of the algorithm is to recognize and pull the content (i.e., the objects and images) from a designated "Content Image", recognize and pull the style (i.e., the line styles, color ranges, brush strokes, cubist patterns, et cetera) from a different designated "Style Image", then combine the two to create a never-before-seen "Generated Image".

# Example
In this example, the Content Image is a photo of me and my wife in Sedona, Arizona.

![Lawton Nicole Small](https://user-images.githubusercontent.com/80790548/142705179-17e9e610-a224-4d90-9dee-98623735bc4b.jpg)

The Style Image is a picture no one can seem to identify (including Google Images), but is nevertheless used in numerous Neural Style Transfer examples.

![Mosaic Small](https://user-images.githubusercontent.com/80790548/142705190-0d40b779-5ea4-40a0-b163-6330c53a2aab.jpg)

The Generated Image has been constructed in such a way that both the content and the style are immediately recognizable.

![LN Mosaic 15 Small](https://user-images.githubusercontent.com/80790548/142705197-c414975f-dc66-4f6e-b533-5e7d5a5301f3.jpg)

# Ingredients
There are four main ingredients when creating a new image using Neural Style Transfer:

1) The Data
2) The Architecutre
3) The Loss Function
4) The Optimizer

# Part I: The Data
The data consists of a user-defined Content Image (often - but not always - a photograph), and a user-defined Style Image (often - but not always - a painting or some other work of art). But there is also a third, Generated Image, which is initialized to be nothing more than static at this point.

![Static](https://user-images.githubusercontent.com/80790548/142705544-6075f6b0-b3d1-4ea2-a66b-47a92876eb50.jpg)

# Preprocessing
Some preprocessing of the input images has to be completed as a first step. Images will likely have to be adjusted for size and number of channels, then converted to tensors so they can be recognized by the VGG19 model (used in this project). Additionally, colors are converted from RGB to BGR, to ensure compatability with the model. Deprocessing (near program completion) is just the opposite. Tensors are converted back to images that can be viewed and saved. Keras (also used in this project) has built-in functions for all of this.

# Part II: The Architecture
The VGG19 model is a Convolutional Neural Network (CNN) originally intended for image recognition and classification, and was trained on the 2012 ImageNet database (14,197,122 different images). The CNN is separated into sixteen convolutional layers (shown in green). The other three layers (purple, for a total of nineteen) are not used. So the focus is only on the convolutional layers.

![VGG19 Model](https://user-images.githubusercontent.com/80790548/142706035-b90ac7d9-0c7f-42a0-9895-dca104919928.jpg)

# Feature Maps
Each layer is a collection of image filters, and each filter searches for a particular pattern (or feature) in the input image. The output of a given filter is called a feature map. Each feature map is then taken through an activation function, which decides whether or not a certain feature is present at a given location in the image.

The feature maps generated in the lower layers of the model becomes the input for the higher layers. More filters are then applied, which generates more feature maps, which become more and more abstract as progress continues through the model.

The filters in the lower layers of the model search for low-level features in the image, such as lines, corners, and blobs.

![Screenshot 1 Small](https://user-images.githubusercontent.com/80790548/142706834-3b2b3b08-23c9-47f5-9c4e-660b67c4b391.jpg)

Using the lower-level feature maps as input, the middle layers might search for more specific objects like eyes, ears, or noses, while the higher layers (using the middle-layer feature maps as input) might search for actual faces (or groups of faces).

![Screenshot 2 Small](https://user-images.githubusercontent.com/80790548/142706843-76e319b3-2b61-4080-8a30-2c7acc536d74.jpg)

In this way, it makes no difference where a face is located in the original image (or even if the face is upside-down).

Had the photo in this example been a photo of a bicycle instead, then the middle layers would have discovered wheels, gears, and handlebars, while the higher layers would have discovered the complete bicycle.

![Bicycle](https://user-images.githubusercontent.com/80790548/142775211-382e8329-7b4d-4230-bccd-42c44cc2493e.jpg)

Jason Yosinski has put together a wonderful (and surprisingly short) YouTube [video](https://www.youtube.com/watch?v=AgkfIQ4IGaM) describing the inner workings of a Convolutional Neural Network, which explains all of this in more detail.

# Capturing the Content and Style
While the content of an image is taken from the higher layers of the model, the style of an image is taken from EVERY layer in the model, and is defined by the correlations between the filter responses (known as the gram matrix). In this way the texture information, line styles, color palettes, et cetera is captured without capturing the global arrangement of the objects in the image.

# Part III: The Loss Function
The eventual output image is created by simultaneously trying to match the content representation of the photograph and the style representation of the painting (or other artwork). There is no "perfect" image that matches both constraints at the same time.  But by minimizing both the content loss and the style loss (collectively, the total loss) as much as possible, the photograph is recreated in the style of the artwork.

![Formula 01](https://user-images.githubusercontent.com/80790548/142708358-a9cee1a4-2d0b-4f12-8dfd-ae4136d97337.jpg)

In this formula, the total loss is minimized using back propagation and traditional optimization functions. Both the content weight and the style weight are pre-determined constants (normally, when training a neural network, the weights are constantly updated. In Neural Style Transfer, the weights remain constant while the generated image itself is constantly updated).

# The Content Loss
Different feature maps in the higher model layers are activated by the presence of different objects. So if two images (the content image and the generated image) have the same (or similar) content, they should have the same (or similar) activations.  The mean squared error between these two activations is the content loss, which is included in the total loss.

# The Style Loss
Calculating the style loss requires more work. Suppose a filter in a lower layer detects a particular texture in an image, while a second filter detects a particular color. If the two are correlated, then the presence of that texture also means the presence of that color. The existence of one depends on the existence of the other. In the higher layers of the model, these two features tend to occur (or not occur) together.

The dot product measure the relationship between two such features. The lesser the product, the weaker the correlation between them, while the greater the product, the stronger the correlation. This gives information about an image's style, and zero information about its spatial structure.

All the feature vectors in a layer are combined together to create a gram matrix.

# The Gram Matrix
The gram matrix is calculated by converting a given vector into a matrix, then multiplying that matrix by its own transpose. For example, the vector V = [1, 2, 3, 4] would be converted into the matrix:

![Screenshot 3](https://user-images.githubusercontent.com/80790548/142708681-bdc8cd91-37fe-4554-ab6c-c855a1dd4119.jpg)

The transpose of that matrix would be:

![Screenshot 4](https://user-images.githubusercontent.com/80790548/142708688-1719e89f-d927-4646-8483-004fa659c0e5.jpg)

And the multiplication of those two would be:

![Screenshot 5](https://user-images.githubusercontent.com/80790548/142708696-82e5d99d-51e7-467b-9efc-4ac977dd4b63.jpg)

# Completing the Formula
The gram matrix is used to capture the distribution of features in an image. The mean squared error between two matrices is used to calculate the style loss, which is then included in the total loss

# Part IV: The Optimizer
The total loss function is dependent on the Generated Image, and the optimizer tells us how to change the Generated Image to make the loss a bit smaller.

The Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer, used in both the original Neural Style Transfer paper by Gatys, et al. and in this project, is computationally more expensive than others, but produces better visual results.

It is a type of second-order optimization algorithm, meaning that it makes use of the second-order derivative of an objective function and belongs to a class of algorithms referred to as Quasi-Newton methods that approximate the second derivative (called the Hessian) for optimization problems where the second derivative cannot be calculated.

# Performance
In one experiment, it out-performed five other optimizers when running 100 iterations over two 300x300 pixel images.  This experiment (and several others) can be found on Slav Ivanov's [website](https://blog.slavv.com/picking-an-optimizer-for-style-transfer-86e7b8cba84b).

![Optimizers](https://user-images.githubusercontent.com/80790548/142727818-60bcaded-5185-45d2-98f6-592ce49a93c9.jpg)

# Broyden, Fletcher, Goldfarb, and Shanno.
During my research, I happened across this photo of Broyden, Fletcher, Goldfarb, and Shanno.

![BFGS](https://user-images.githubusercontent.com/80790548/142742617-af5667aa-4cdf-43ea-a0c7-b98341365486.jpg)

# The Basics
Those are the absolute basics for generating an image using Neural Style Transfer.  But there are (at least) two import downsides to the basic algorithm:

The first is the poor quality of the Generated Images.
The second is the processing time required.

# Image Quality
It was soon discovered that minimizing ONLY the content and style losses led to highly pixelated and "noisy" images.  To continue the example:

These two images show the horizontal and vertical deltas in the original Content Image...

![Horizontal Content Small](https://user-images.githubusercontent.com/80790548/142709046-a47643b2-6895-4e32-921a-1705dd4bdd86.jpg)     ![Vertical Content Small](https://user-images.githubusercontent.com/80790548/142709048-2d5379f9-a3f6-4627-9457-0cce26567e16.jpg)

...while these two images show the horizontal and vertical deltas in the original Generated Image.

![Horizontal Generated Small](https://user-images.githubusercontent.com/80790548/142709057-14199294-1144-413e-b6c4-f2dd4104213d.jpg)     ![Vertical Generated Small](https://user-images.githubusercontent.com/80790548/142709075-ee930067-b4a8-4eae-87f0-6489bc603ac0.jpg)

# Total Variation Loss

To correct this problem, Total Variation Loss (which can be conceptualized as "edge detection") was introduced to ensure spatial quality and "smoothness" in the Generated Image.

The image on the left does not use Total Variation Loss, while the image on the right does. The difference between the two is evident.

![LN Mosaic 15 No TVL Small](https://user-images.githubusercontent.com/80790548/142735399-5b93c2d4-97d6-4079-8bca-a449b9db92cc.jpg)     ![LN Mosaic 15 Small](https://user-images.githubusercontent.com/80790548/142735366-0bb7251f-7459-431a-aa4c-4bab749af465.jpg)

# Processing Time
The second important downside to this basic algorithm is the processing time. This image took between 6 and 7 hours to create on an Intel i3 CPU with 32GB RAM. This renders the entire application nearly useless as a web application. 

![LN Mosaic 100](https://user-images.githubusercontent.com/80790548/142730000-8e7260da-0993-46db-aebd-256247c323ca.jpg)

# TF-Lite

For the web application part of this project, I used a different model called TensorFlow Lite (or TF-Lite). This model was trained on roughly 80,000 paintings and is able to generalize on paintings previously unobserved. What it lacks in visual quality it makes up for in speed, and is therefore used in several online tools.

# Speed Versus Quality
The image on the left was created using TensorFlow/Keras, and is shown for comparison.
The center image was created by the web application version of this project, which uses TF-Lite.
The overall image quality is not as good, but the image was created in just over one minute.
The image on the right was generated by someone else’s online tool, also using TF-Lite, and was generated in just a few seconds.

![LN Mosaic 100 Small](https://user-images.githubusercontent.com/80790548/142775824-14a6bc2f-77fb-4371-bc12-1f835cbdc21a.jpg)     ![TensorFlow Hub Small](https://user-images.githubusercontent.com/80790548/142775862-605f68d2-badb-456b-ac05-40fef684d065.jpg)     ![Online Tool](https://user-images.githubusercontent.com/80790548/142775827-9d15616f-31c4-4041-9e64-0150418971e2.jpg)

# The Obligatory Picture Gallery

"Self-Portrait" by Pablo Picasso

![Picasso Small](https://user-images.githubusercontent.com/80790548/142733423-0523928e-740e-491d-b93d-7902f5a2248a.jpg)   ![LN Picasso 25 Small](https://user-images.githubusercontent.com/80790548/142733389-7c5e09c9-16cb-409b-8147-7ef301bdcc09.jpg)

"Yellow Red Blue" by Wassily Kandinsky

![Kandinsky Small](https://user-images.githubusercontent.com/80790548/142742073-7429e0e5-80e4-4655-8c92-23d6c38a6135.jpg)     ![LN Kandinsky 15 Small](https://user-images.githubusercontent.com/80790548/142742042-0c42060c-0535-4aaf-add4-ee76c0551960.jpg)

"The Great Wave Off Kanagawa" by Katsushika Hokusai

![Kanagawa Small](https://user-images.githubusercontent.com/80790548/142739600-19703d20-e33a-4cd5-a0ab-c5987cf5aea2.jpg)     ![LN Kanagawa 15 Small](https://user-images.githubusercontent.com/80790548/142739557-fcff867e-c1ad-4584-a554-db2241342b3c.jpg)

"Couple Under One Umbrella" by Leonid Afrimov

![Aframov Small](https://user-images.githubusercontent.com/80790548/142743038-714da528-2567-41b9-abec-27670626931c.jpg)     ![LN Aframov 15 Small](https://user-images.githubusercontent.com/80790548/142743050-feb47568-7b48-4d50-ac7d-ae8d9bdee856.jpg)

"A Conspiracy Eternal" by Skinner

![Skinner Small](https://user-images.githubusercontent.com/80790548/142740774-df08abe3-04b9-472c-ab51-b1ac29a17aa2.jpg)     ![LN Skinner 15 Small](https://user-images.githubusercontent.com/80790548/142740740-e2af717c-a683-4579-8a61-9726b4a18243.jpg)


# Links
1) The original paper, "A Neural Algorithm of Artistic Style" by Gatys, Ecker, and Bethge can be found [here](https://arxiv.org/abs/1508.06576).
2) Leon Gatys released a Jupyter Notebook showing his PyTorch implementation of his original paper, which can be found [here](https://github.com/leongatys/PytorchNeuralStyleTransfer).
3) "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", by Justin Johnson, Alexandre Alahi, and Fei-Fei Li, is a more performance-based take on Neural Style Transfer. In this paper, which can be found [here](https://cs.stanford.edu/people/jcjohns/eccv16/), the authors claim similar qualitative results, but three orders of magnitude faster.
4) Justin Johnson has also released code on Github, which can be found [here](https://github.com/jcjohnson/fast-neural-style).
5) The ImageNet database can be found [here](image-net.org).

# The Code

This repository includes both a Jupyter notebook version, and a PyCharm version.

The Jupyter Notebook version produces better visual results, but CAN take several hours to run  - depending on how you adjust the parameters.

The PyCharm version utilizes TensorFlow Hub, which produces less-appealing images overall, but does so in less than a minute.  It was turned into a web application using a Flask interface, which was then turned into a Docker image (which is located on the Docker hub at lawton12345/the_lawton_project). This version requires jpg, jpeg, or png images (2MB or smaller) as input.

# Jupyter
Requirements:
1) Access to (and working knowledge of) Jupyter Notebooks.
2) Python 3.8
3) TensorFlow 2.7
4) Numpy 1.20

To Run the Program:
1) Download the capstone.ipynb file and add it into your own Jupyter Notebook.
2) Adjust the parameters (cell 3) and filenames (cell 5) to your liking.

# PyCharm
Requirements:
1) Docker Desktop installed on your local machine.
2) A working knowledge of Docker and Windows PowerShell (or whatever tool you use to run Docker commands).

To Run the Program:
1) Pull the_lawton_project from lawton12345 on the Docker hub.
2) Open Windows PowerShell.
3) Enter the command:  docker run -p 5000:5000 the_lawton_project
4) Open a webpage to //localhost:5000
6) On the opening page, select the content image and the style image separately.
   Both files must have either .JPG, .JPEG, or .PNG extensions, and both files must be 2MB or smaller.

![Index1 Screen](https://user-images.githubusercontent.com/80790548/135180618-e9b2ae2e-cb61-42c1-9f09-dc58fe71830a.jpg)


7) After you have selected both files, click the Upload button.
   Both file names should be clearly visible before you continue.

![Index2 Screen](https://user-images.githubusercontent.com/80790548/135180628-f4da7de7-045c-4a7f-931a-1aa8511dc925.jpg)


8) On the next page, both files should show up in the preview.

![Upload Screen](https://user-images.githubusercontent.com/80790548/135180369-b983b27b-13e5-45eb-bb8a-51bbd9da6798.jpg)

9) Click the "Run Style Transfer" button when you are satisfied with the preview.
   The generated image will then be displayed on the following page.

![Result Screen](https://user-images.githubusercontent.com/80790548/135180425-cc2362be-6987-4962-9ecc-438a484c38a7.jpg)


10) Close both the web page and Windows PowerShell when you have finished experimenting.

# What is Next?
Aleksa Gordić has put together a fun [video](https://www.youtube.com/watch?v=B22nIUhXo4E) on YouTube, in which he applies Neural Style Transfer to videos instead of still images. This is the next project I'd like to take on.

