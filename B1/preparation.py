"""
Data preprocessing operations.
"""
# Import packages
import os
import shutil
import sys
import PIL
import glob
import dlib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


# def get_target_size(cartoon_set_img_dir):
#     image = PIL.Image.open(cartoon_set_img_dir + os.listdir(cartoon_set_img_dir)[0])
#     targe_size = image.size
#     return targe_size


# def get_input_shape(cartoon_set_img_dir, channels=3):
#     image = PIL.Image.open(cartoon_set_img_dir + os.listdir(cartoon_set_img_dir)[0])
#     height, width = image.size
#     input_shape = (width, height, channels)
#     return input_shape


def load_data(cartoon_set_df):
    face_shape_df = cartoon_set_df.copy()
    face_shape_df.drop(face_shape_df.columns[0], axis=1, inplace=True)
    face_shape_df.drop(["eye_color"], axis=1, inplace=True)
    face_shape_train_df, face_shape_test_df = train_test_split(
        face_shape_df, test_size=0.2, random_state=66
    )
    return face_shape_train_df, face_shape_test_df


def process_data(
    b1_dir, cartoon_set_img_dir, face_shape_df, usage, feature, class_mapper_dict
):
    # predictor_path = "shape_predictor_68_face_landmarks.dat"
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(os.path.join(b1_dir, predictor_path))
    face_shape_df_faceless = face_shape_df.copy()
    face_shape_df_faceless["faceless"] = np.nan
    # Path to classifier
    hcasc_eye_path = os.path.join(b1_dir, "haarcascade_eye.xml")
    # Instantiate eye detection
    eye_classifier = cv2.CascadeClassifier(hcasc_eye_path)
    for i in tqdm(
        sorted(face_shape_df.face_shape.unique()),
        desc="Mapping Classified Cartoon Set Images To Temp Train/Test Directories",
    ):
        print("Creating Directory For Class {0}".format(class_mapper_dict[str(i)]))
        current_face_shape_output_path = os.path.join(
            b1_dir, str(usage), class_mapper_dict[str(i)]
        )
        if os.path.exists(current_face_shape_output_path) and os.path.isdir(
            current_face_shape_output_path
        ):
            shutil.rmtree(current_face_shape_output_path)
        try:
            os.makedirs(current_face_shape_output_path)
        except OSError:
            print("Failed To Create Directory %s" % current_face_shape_output_path)
        else:
            print("Created Directory %s " % current_face_shape_output_path)
        is_current_face_shape = face_shape_df["face_shape"] == str(i)
        current_face_shape = face_shape_df[is_current_face_shape]
        with tqdm(total=current_face_shape.shape[0]) as pbar:
            pbar.set_description(
                "Processing Images For Class %s" % class_mapper_dict[str(i)]
            )
            for index, im in current_face_shape.iterrows():
                im_path = os.path.join(cartoon_set_img_dir, im["file_name"])
                img = cv2.imread(im_path)
                img = img[290:450, 170:330]
                cv2.imwrite(
                    os.path.join(current_face_shape_output_path, im["file_name"]),
                    img,
                )
                pbar.update(1)
                # dets = detector(img, 1)

                # if len(dets) > 0:
                #     for k, d in enumerate(dets):
                #         # print(
                #         #     "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #         #         k, d.left(), d.top(), d.right(), d.bottom()
                #         #     )
                #         # )
                #         # Get the landmarks/parts for the face in box d.
                #         shape = predictor(img, d)
                #         # The next lines of code just get the coordinates for the mouth
                #         # and crop the mouth from the image.This part can probably be optimised
                #         # by taking only the outer most points.
                #         xmouthpoints = [shape.part(x).x for x in range(3, 15)]
                #         ymouthpoints = [shape.part(x).y for x in range(3, 15)]
                #         maxx = max(xmouthpoints)
                #         minx = min(xmouthpoints)
                #         maxy = max(ymouthpoints)
                #         miny = min(ymouthpoints)

                #         # to show the mouth properly pad both sides
                #         pad = 10
                #         # basename gets the name of the file with it's extension
                #         # splitext splits the extension and the filename
                #         # This does not consider the condition when there are multiple faces in each image.
                #         # if there are then it just overwrites each image and show only the last image.
                #         # filename = os.path.splitext(os.path.basename(f))[0]

                #         crop_image = img[
                #             miny - pad : maxy + pad, minx - pad : maxx + pad
                #         ]
                #         # The mouth images are saved in the format 'mouth1.jpg, mouth2.jpg,..
                #         # Change the folder if you want to. They are stored in the current directory
                #         cv2.imwrite(
                #             os.path.join(
                #                 current_face_shape_output_path, im["file_name"]
                #             ),
                #             crop_image,
                #         )
                # else:
                #     face_shape_df_faceless.loc[index, "faceless"] = True
                # pbar.update(1)
    # valid = face_shape_df_faceless[face_shape_df_faceless.faceless != True]
    # invalid = face_shape_df_faceless[face_shape_df_faceless.faceless == True]
    # perc_valid = str(valid.shape[0] * 100 / face_shape_df.shape[0]) + "%"
    # print(perc_valid + " " + "of the dataset provided have detectable faces.")
    # perc_reduction = (
    #     str(round(100 - valid.shape[0] * 100 / face_shape_df.shape[0], 2)) + "%"
    # )
    # print(
    #     perc_reduction
    #     + " "
    #     + "of the dataset provided will therefore not be processed."
    # )
    # return valid, invalid


def get_img_array(cartoon_set_img_dir):
    img_array = []
    img_dir = glob.glob(os.path.join(cartoon_set_img_dir, "*.png"))
    for img_file in img_dir:
        img = cv2.imread(img_file)
        img_array.append(img)
    return img_array


# def gen_data(cartoon_set_img_dir, b1_dir):
#     train_imgen = ImageDataGenerator(
#         validation_split=0.25,
#         rescale=1.0 / 255,
#         width_shift_range=[-0.10, 0.10],
#         height_shift_range=[-0.10, 0.10],
#         horizontal_flip=True,
#         rotation_range=10,
#         zoom_range=[0.90, 1.10],
#     )
#     # print("Fitting Train ImageGenerator")
#     test_imgen = ImageDataGenerator(rescale=1.0 / 255)
#     # print("Fitting Test ImageGenerator")
#     face_shape_train_flow = train_imgen.flow_from_directory(
#         os.path.join(b1_dir, "train"),
#         color_mode="rgb",
#         class_mode="categorical",
#         # Should be (218, 178)
#         target_size=get_target_size(cartoon_set_img_dir),
#         batch_size=32,
#         shuffle=True,
#         seed=66,
#         subset="training",
#     )
#     face_shape_val_flow = train_imgen.flow_from_directory(
#         os.path.join(b1_dir, "train"),
#         color_mode="rgb",
#         class_mode="categorical",
#         target_size=get_target_size(cartoon_set_img_dir),
#         batch_size=32,
#         shuffle=True,
#         seed=66,
#         subset="validation",
#     )
#     face_shape_test_flow = test_imgen.flow_from_directory(
#         os.path.join(b1_dir, "test"),
#         color_mode="rgb",
#         class_mode="categorical",
#         target_size=get_target_size(cartoon_set_img_dir),
#         batch_size=32,
#         shuffle=True,
#         seed=66,
#     )
#     return face_shape_train_flow, face_shape_val_flow, face_shape_test_flow
