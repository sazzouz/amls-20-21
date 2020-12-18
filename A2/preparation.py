"""
Data preprocessing operations.
"""
# Import packages
import os
import shutil
import sys
import PIL
import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import dlib


# def get_target_size(celeba_img_dir):
#     image = PIL.Image.open(celeba_img_dir + os.listdir(celeba_img_dir)[0])
#     targe_size = image.size
#     return targe_size


# def get_input_shape(celeba_img_dir, channels=3):
#     image = PIL.Image.open(celeba_img_dir + os.listdir(celeba_img_dir)[0])
#     height, width = image.size
#     input_shape = (width, height, channels)
#     return input_shape


def load_data(celeba_df):
    smiling_df = celeba_df.copy()
    smiling_df.drop(smiling_df.columns[0], axis=1, inplace=True)
    smiling_df.drop(["gender"], axis=1, inplace=True)
    smiling_train_df, smiling_test_df = train_test_split(
        smiling_df, test_size=0.2, random_state=66
    )
    return smiling_train_df, smiling_test_df


# def get_img_array(celeba_img_dir):
#     img_array = []
#     img_dir = glob.glob(os.path.join(celeba_img_dir, "*.jpg"))
#     for img_file in img_dir:
#         img = cv2.imread(img_file)
#         img_array.append(img)
#     return img_array


def process_data(a2_dir, celeba_img_dir, smiling_df, usage, feature, class_mapper_dict):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(a2_dir, predictor_path))
    smiling_df_mouthless = smiling_df.copy()
    smiling_df_mouthless["mouthless"] = np.nan
    for i in tqdm(
        sorted(smiling_df.smiling.unique()),
        desc="Mapping Classified Celeba Images To Temp Train/Test Directories",
    ):
        print("Creating Directory For Class {0}".format(class_mapper_dict[str(i)]))
        current_smiling_output_path = os.path.join(
            a2_dir, str(usage), class_mapper_dict[str(i)]
        )
        if os.path.exists(current_smiling_output_path) and os.path.isdir(
            current_smiling_output_path
        ):
            shutil.rmtree(current_smiling_output_path)
        try:
            os.makedirs(current_smiling_output_path)
        except OSError:
            print("Failed To Create Directory %s" % current_smiling_output_path)
        else:
            print("Created Directory %s " % current_smiling_output_path)
        is_current_smiling = smiling_df[str(feature)] == str(i)
        current_smiling = smiling_df[is_current_smiling]
        with tqdm(total=current_smiling.shape[0]) as pbar:
            pbar.set_description(
                "Processing Images For Class %s" % class_mapper_dict[str(i)]
            )
            for index, im in current_smiling.iterrows():
                im_path = os.path.join(celeba_img_dir, im["img_name"])
                img = cv2.imread(im_path)
                dets = detector(img, 1)
                if len(dets) > 0:
                    for k, d in enumerate(dets):
                        # print(
                        #     "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                        #         k, d.left(), d.top(), d.right(), d.bottom()
                        #     )
                        # )
                        # Get the landmarks/parts for the face in box d.
                        shape = predictor(img, d)
                        # The next lines of code just get the coordinates for the mouth
                        # and crop the mouth from the image.This part can probably be optimised
                        # by taking only the outer most points.
                        xmouthpoints = [shape.part(x).x for x in range(48, 67)]
                        ymouthpoints = [shape.part(x).y for x in range(48, 67)]
                        maxx = max(xmouthpoints)
                        minx = min(xmouthpoints)
                        maxy = max(ymouthpoints)
                        miny = min(ymouthpoints)

                        # to show the mouth properly pad both sides
                        pad = 10
                        # basename gets the name of the file with it's extension
                        # splitext splits the extension and the filename
                        # This does not consider the condition when there are multiple faces in each image.
                        # if there are then it just overwrites each image and show only the last image.
                        # filename = os.path.splitext(os.path.basename(f))[0]

                        crop_image = img[
                            miny - pad : maxy + pad, minx - pad : maxx + pad
                        ]
                        # The mouth images are saved in the format 'mouth1.jpg, mouth2.jpg,..
                        # Change the folder if you want to. They are stored in the current directory
                        cv2.imwrite(
                            os.path.join(current_smiling_output_path, im["img_name"]),
                            crop_image,
                        )
                else:
                    smiling_df_mouthless.loc[index, "mouthless"] = True
                pbar.update(1)
    valid = smiling_df_mouthless[smiling_df_mouthless.mouthless != True]
    invalid = smiling_df_mouthless[smiling_df_mouthless.mouthless == True]
    perc_valid = str(valid.shape[0] * 100 / smiling_df.shape[0]) + "%"
    print(perc_valid + " " + "of the dataset provided have detectable mouths.")
    perc_reduction = (
        str(round(100 - valid.shape[0] * 100 / smiling_df.shape[0], 2)) + "%"
    )
    print(
        perc_reduction
        + " "
        + "of the dataset provided will therefore not be processed."
    )
    return valid, invalid


# def gen_data(celeba_img_dir, a2_dir, img_array):
#     train_imgen = ImageDataGenerator(
#         validation_split=0.25,
#         rescale=1.0 / 255,
#         width_shift_range=[-0.10, 0.10],
#         height_shift_range=[-0.10, 0.10],
#         horizontal_flip=True,
#         rotation_range=10,
#         zoom_range=[0.90, 1.10],
#     )
#     print("Fitting Train ImageGenerator")
#     train_imgen.fit(img_array)
#     test_imgen = ImageDataGenerator(rescale=1.0 / 255)
#     print("Fitting Test ImageGenerator")
#     test_imgen.fit(img_array)
#     smiling_train_flow = train_imgen.flow_from_directory(
#         os.path.join(a2_dir, "train"),
#         color_mode="rgb",
#         class_mode="categorical",
#         # Should be (218, 178)
#         target_size=get_target_size(celeba_img_dir),
#         batch_size=32,
#         shuffle=True,
#         seed=66,
#         subset="training",
#     )
#     smiling_val_flow = train_imgen.flow_from_directory(
#         os.path.join(a2_dir, "train"),
#         color_mode="rgb",
#         class_mode="categorical",
#         target_size=get_target_size(celeba_img_dir),
#         batch_size=32,
#         shuffle=False,
#         seed=66,
#         subset="validation",
#     )
#     smiling_test_flow = test_imgen.flow_from_directory(
#         os.path.join(a2_dir, "test"),
#         color_mode="rgb",
#         class_mode="categorical",
#         target_size=get_target_size(celeba_img_dir),
#         batch_size=32,
#         shuffle=False,
#         seed=66,
#     )
#     return smiling_train_flow, smiling_val_flow, smiling_test_flow
