# Tool-image-preparation-deep-learning

Machine Learning applies machine learning algorithms to learn from data. It is critical that you
feed them the right data for the problem you want to solve. Even if you have good data, you need
to make sure that it is on a useful scale, format and even that essential features are included. To
get a better result, it is recommended to look through your data, see the quality and quantity of
data, work to improve the quality of images and/or apply augmentation to increase your dataset or
remove unwanted images. The process for getting data ready for a machine learning algorithmcan be
summarized in three steps:
1. Data collection and selection
2. Image preprocessing
3. Image augmentation

## Data collection and selection

This step is about selecting a subset of all available images that you will work with. We have a strong
desire to include all available data to train our algorithmbut, this might not always be true. We have to
consider data that are needed to tackle our problem. Therefore, you have to know what data you have,
what data is not available that you want to have it, exclude the data that you do not need to address
the problem under consideration.

## Image preprocessing

After selecting the appropriate images, we need to think how we are going to use our data with the
proposed technique. The type of the preprocessing to be applied strongly depends on the type of the
data and the machine learning tool you chose. Themost commonly applied image preprocessing in
machine learning and specifically to deep learning includes image contract enhancement, sampling
and image denoising. The data already collectedmight not be in the format that is suitable for your
machine learning algorithm. Here format refers to the data type of the images or image normalization.
Medical imaging devices are susceptible to noise. Image denoising is the process to remove the
noise from the image naturally corrupted by the noise. In addition to that, there may be far more
selected data available than you need to work with. More data can result in much longer running
times for algorithms and larger computational and memory requirements. Thus you need to sample
some more appropriated data fromthe images.

Another most important steps in deep learning and other machine learning are scaling and patch
generation. The gathered data might have different size, so it is needed to scale to the same shape
before feeding to your learning algorithm. When we train with deep learning we have to consider
memory consumption as we have limited GPU memory, and we have to use a batch of images to have
a more generalized model. Thus, for 3D volume images and pathology images, it is not possible to
train whole volume or whole slide with mini-batch size. In this case, we have to extract patches and apply reconstruction to get the final result. Moreover, extracting patches is used to increase training
data as annotated medical image are scarce.

## Image Augmentation

Image augmentation refers to applying deformation or transformation on the available dataset. By
augmenting the images, you are training the network not to overfit your dataset with regards to the
type of augmentation. For example, rotating an image in various angles will make the model to
be invariant to rotation of the objects in the images. The most commonly applied augmentation
techniques are zooming, scaling, rotation or adding random noise. So although new information
is not added into the network, the synthetic data augmentation added into the network can both
improve the results attained from the network and allow for training with less data, which is useful
in medical application. If there are features extracted before applying augmenting the image and
if the sensitive to the deformation applied, the same deformation should be applied.However, it is
important to note that augmentation is only useful when semantically correct.

Although image preprocessing, patch extraction, and image contrast enhancement are determinant
when we apply machine learning specifically deep learning inmedical image processing and analysis,
to the best of our knowledge, there is no a software for extracting patches, and applying image
augmentation with different ground truth types(landmark file for example .xml files, ground truth or
mask images). The main goal this project is to develop a software that contains the most commonly
used image preprocessing and augmentation techniques to apply machine learning and deep learning
in amedical image taking into consideration different ground truth types.

This report is organized as follows. In section two, we briefly explain datasets used to test our software.
In section three we describe our software component and how to use the software. Result and discussion
are presented in section four. Finally, project management and conclusion are presented.
