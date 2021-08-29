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


def load_data(celeba_df):
    gender_df = celeba_df.copy()
    gender_df.drop(gender_df.columns[0], axis=1, inplace=True)
    gender_df.drop(["smiling"], axis=1, inplace=True)
    gender_train_df, gender_test_df = train_test_split(
        gender_df, test_size=0.2, random_state=66
    )
    return gender_train_df, gender_test_df


def process_data(a1_dir, celeba_img_dir, gender_df, usage, feature, class_mapper_dict):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(a1_dir, predictor_path))
    print("\nPreparing data\n")
    gender_df_faceless = gender_df.copy()
    gender_df_faceless["faceless"] = np.nan
    for i in tqdm(
        sorted(gender_df.gender.unique()),
        desc="Mapping Classified Celeba Images To Temp Train/Test Directories",
    ):
        print("Creating Directory For Class {0}".format(class_mapper_dict[str(i)]))
        current_gender_output_path = os.path.join(
            a1_dir, str(usage), class_mapper_dict[str(i)]
        )
        if os.path.exists(current_gender_output_path) and os.path.isdir(
            current_gender_output_path
        ):
            shutil.rmtree(current_gender_output_path)
        try:
            os.makedirs(current_gender_output_path)
        except OSError:
            print("Failed To Create Directory %s" % current_gender_output_path)
        else:
            print("Created Directory %s " % current_gender_output_path)
        is_current_gender = gender_df[str(feature)] == str(i)
        current_gender = gender_df[is_current_gender]
        with tqdm(total=current_gender.shape[0]) as pbar:
            pbar.set_description(
                "Processing Images For Class %s" % class_mapper_dict[str(i)]
            )
            for index, im in current_gender.iterrows():
                im_path = os.path.join(celeba_img_dir, im["img_name"])
                img = cv2.imread(im_path)
                dets = detector(img, 1)
                if len(dets) > 0:
                    for k, d in enumerate(dets):
                        shape = predictor(img, d)
                        xmouthpoints = [shape.part(x).x for x in range(1, 67)]
                        ymouthpoints = [shape.part(x).y for x in range(1, 67)]
                        maxx = max(xmouthpoints)
                        minx = min(xmouthpoints)
                        maxy = max(ymouthpoints)
                        miny = min(ymouthpoints)
                        pad = 0

                        crop_image = img[
                            miny - pad : maxy + pad, minx - pad : maxx + pad
                        ]

                        cv2.imwrite(
                            os.path.join(current_gender_output_path, im["img_name"]),
                            crop_image,
                        )
                else:
                    gender_df_faceless.loc[index, "faceless"] = True
                pbar.update(1)
    # Isolate images which have no detectable region of interest, proceed with 'valid' images
    valid = gender_df_faceless[gender_df_faceless.faceless != True]
    invalid = gender_df_faceless[gender_df_faceless.faceless == True]
    # Determine proportion of valid images
    perc_valid = str(valid.shape[0] * 100 / gender_df.shape[0]) + "%"
    print(perc_valid + " " + "of the dataset provided have detectable faces.")
    perc_reduction = (
        str(round(100 - valid.shape[0] * 100 / gender_df.shape[0], 2)) + "%"
    )
    print(
        perc_reduction
        + " "
        + "of the dataset provided will therefore not be processed."
    )
    return valid, invalid