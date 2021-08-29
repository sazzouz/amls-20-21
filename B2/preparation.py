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

def load_data(cartoon_set_df):
    eye_color_df = cartoon_set_df.copy()
    eye_color_df.drop(eye_color_df.columns[0], axis=1, inplace=True)
    eye_color_df.drop(["face_shape"], axis=1, inplace=True)
    eye_color_train_df, eye_color_test_df = train_test_split(
        eye_color_df, test_size=0.2, random_state=66
    )
    return eye_color_train_df, eye_color_test_df


def process_data(
    b2_dir, cartoon_set_img_dir, eye_color_df, usage, feature, class_mapper_dict
):
    eye_color_sunglasses_df = eye_color_df.copy()
    eye_color_sunglasses_df["sunglasses"] = np.nan
    # Path to classifier
    hcasc_eye_path = os.path.join(b2_dir, "haarcascade_eye.xml")
    # Instantiate eye detection
    eye_classifier = cv2.CascadeClassifier(hcasc_eye_path)
    for i in tqdm(
        sorted(eye_color_df.eye_color.unique()),
        desc="Mapping Classified Cartoon Set Images To Temp Train/Test Directories",
    ):
        print("Creating Directory For Class {0}".format(class_mapper_dict[str(i)]))
        current_eye_color_output_path = os.path.join(
            b2_dir, str(usage), class_mapper_dict[str(i)]
        )
        current_sunglasses_output_path = os.path.join(b2_dir, str(usage), "sunglasses")
        if os.path.exists(current_eye_color_output_path) and os.path.isdir(
            current_eye_color_output_path
        ):
            shutil.rmtree(current_eye_color_output_path)
        try:
            os.makedirs(current_eye_color_output_path)
        except OSError:
            print("Failed To Create Directory %s" % current_eye_color_output_path)
        else:
            print("Created Directory %s " % current_eye_color_output_path)
        is_current_eye_color = eye_color_df["eye_color"] == str(i)
        current_eye_color = eye_color_df[is_current_eye_color]
        with tqdm(total=current_eye_color.shape[0]) as pbar:
            pbar.set_description(
                "Processing Images For Class %s" % class_mapper_dict[str(i)]
            )
            for index, im in current_eye_color.iterrows():
                im_path = os.path.join(cartoon_set_img_dir, im["file_name"])
                img = cv2.imread(im_path)
                get_eyes = eye_classifier.detectMultiScale(img)
                if list(get_eyes):
                    # Apply crop

                    img = img[240:285, 180:320]
                    cv2.imwrite(
                        os.path.join(current_eye_color_output_path, im["file_name"]),
                        img,
                    )
                else:
                    eye_color_sunglasses_df.loc[index, "sunglasses"] = True
                    if not os.path.exists(current_sunglasses_output_path):
                        try:
                            os.makedirs(current_sunglasses_output_path)
                        except OSError:
                            print(
                                "Failed To Create Directory %s"
                                % current_sunglasses_output_path
                            )
                        else:
                            print(
                                "Created Directory %s " % current_sunglasses_output_path
                            )
                    elif os.path.isdir(current_sunglasses_output_path):
                        # Apply crop

                        img = img[240:285, 180:320]
                        cv2.imwrite(
                            os.path.join(
                                current_sunglasses_output_path, im["file_name"]
                            ),
                            img,
                        )
                pbar.update(1)
    # Isolate images which have no detectable region of interest, proceed with 'valid' images
    valid = eye_color_sunglasses_df[eye_color_sunglasses_df.sunglasses != True]
    invalid = eye_color_sunglasses_df[eye_color_sunglasses_df.sunglasses == True]
    # Determine proportion of valid images
    perc_valid = str(valid.shape[0] * 100 / eye_color_df.shape[0]) + "%"
    perc_reduction = (
        str(round(100 - valid.shape[0] * 100 / eye_color_df.shape[0], 2)) + "%"
    )
    print(perc_valid + " " + "of the dataset provided have detectable eyes.")
    perc_reduction = (
        str(round(100 - valid.shape[0] * 100 / eye_color_df.shape[0], 2)) + "%"
    )
    print(
        perc_reduction
        + " "
        + "of the dataset provided will therefore be added to new class 'sunglasses'."
    )
    return valid, invalid

