import os
import numpy as np
import cv2
from skimage import io


from Utilits.image_reading_utilits import read_dicom,read_image, read_nii

class OtherPreprocessing:
    """
    OtherProcessing class applies image contrast enhancement techniques
    """
    def __init__(self,
                 input_image = None,
                 ref_image = None,
                 appply_on_batch_files = False,
                 source_folder = None,
                 dst_folder=None,
                 img_name=None,
                 normalize  = False,
                 hist_match  = None,
                 adaptive_histeq=True):

        """
        Class constructor

        :param input_image: Input image
        :param ref_image: reference image for histogram matching
        :param appply_on_batch_files: apply to batch files or to single file
        :param source_folder: image source folder in case of batch processing
        :param dst_folder: saving folder
        :param img_name: input image name
        :param normalize: Flag for binarize
        :param hist_match: Flag histogram matching
        :param adaptive_histeq: Flag for adaprive hist equalization
        """

        # input target iamge
        self.input_image  = input_image

        # reference image for histogram matching
        self.ref_image = ref_image

        # preprocessing applied on a batch of files or single image
        self.appply_on_batch_files  = appply_on_batch_files

        self.image_name  = img_name
        self.source_folder  = source_folder
        self.dst_folder = dst_folder

        self.normalize  = normalize
        self.histo_match_ = hist_match
        self.adaptive_histeq = adaptive_histeq

        if self.appply_on_batch_files is False:
            self.preprocessed_images_folder = os.path.join(self.dst_folder,'Otherpreprocessing')
            if os.path.exists(self.preprocessed_images_folder) is not True:
                os.mkdir(self.preprocessed_images_folder)
        else:
            self.preprocessed_images_folder = os.path.join(self.dst_folder,'Otherpreprocessing')
            if os.path.exists(self.preprocessed_images_folder) is not True:
                os.makedirs(self.preprocessed_images_folder)




    def hist_match(self):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        # taken from https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
        """

        oldshape = self.input_image.shape
        target_image = self.input_image.ravel()
        self.ref_image = self.ref_image.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(self.input_image, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(self.ref_image, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    def histogram_equalization(self):
        """
        apply histogram equalization to an image
        :return: none
        """
        M = np.max(self.input_image)
        m = np.min(self.input_image)
        im_normalized = (self.input_image - m) / M * 255  # .astype()
        im_normalized = im_normalized.astype('uint8')
        equ_im = cv2.equalizeHist(im_normalized)

        return  equ_im
    
    def adaptive_histogram_equalization(self):
        """
        Apply adaptive histogram equalization
        :return:None
        """
        M = np.max(self.input_image)
        m = np.min(self.input_image)
        im_normalized = (self.input_image - m) / M * 255  # .astype()
        im_normalized = im_normalized.astype('uint8')

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        equ_im = clahe.apply(im_normalized)


        return  equ_im

    def intensity_normalization(self):
        """
        intensity normalization
        :return: none
        """
        # apply intensity normalization
        min_  = np.min(self.input_image)
        max_ = np.max(self.input_image - min_).astype('float32')

        return (self.input_image - min_).astype('float32') / max_

    def apply_otherpreprocessing(self):
        """
        this function cheks if it the operation is requested on a single image or to abatch files and call
        the respective function
        :return:
        """

        if self.appply_on_batch_files is False: # single image
            img_name = self.image_name
            img_name = os.path.split(img_name)
            img_name = img_name[-1]
            img_name_split = img_name.split(".")
            self.img_name_ = img_name_split[0]
            self.do_preprocessing()

        else: # apply to all images in the folder
            assert len(os.listdir()), 'source folder is empty'
            files  = os.listdir(self.source_folder)
            for images_name in files:
                self.image_name = images_name
                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                self.img_name_ = img_name_split[0]
                img_extension = img_name_split[-1]
                name_ = os.path.join(self.source_folder,images_name)
                if img_extension == 'dcm':
                    self.input_image = read_dicom(name_)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.input_image = read_nii(name_)
                else:
                    self.input_image = read_image(name_)

                self.do_preprocessing()
        print('All requested preprcessing are done!')

    def do_preprocessing(self):
        """
        check what type of preprocessing is requested and and call the respective function
        :return:
        """
        if self.histo_match_ is True:
            result = self.hist_match()
            min_ = np.min(result)
            max_ = np.max(result - min_)
            result = (result - min_).astype('float')/float(max_) * 255
            img_name = os.path.join(self.preprocessed_images_folder, self.img_name_ + '_hist_match.png')
            io.imsave(img_name, result.astype('uint8'))
        if self.adaptive_histeq is True:
            result = self.histogram_equalization()
            min_ = np.min(result)
            max_ = np.max(result - min_)
            result = (result - min_).astype('float')/float(max_) * 255

            img_name = os.path.join(self.preprocessed_images_folder, self.img_name_ + '_histeq.png')
            io.imsave(img_name, result.astype('uint8'))
        else:
            result = self.adaptive_histogram_equalization()
            min_ = np.min(result)
            max_ = np.max(result - min_)
            result = (result - min_).astype('float')/float(max_) * 255

            img_name = os.path.join(self.preprocessed_images_folder, self.img_name_ + '_ada_histeq.png')
            io.imsave(img_name, result.astype('uint8'))
            
        if self.normalize is True:
            result = self.intensity_normalization()
            min_ = np.min(result)
            max_ = np.max(result - min_)
            result = (result - min_).astype('float')/float(max_) * 255

            img_name = os.path.join(self.preprocessed_images_folder, self.img_name_ + '_norm.png')
            io.imsave(img_name, result.astype('uint8'))

