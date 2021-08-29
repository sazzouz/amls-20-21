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



def load_data(celeba_df):
    smiling_df = celeba_df.copy()
    smiling_df.drop(smiling_df.columns[0], axis=1, inplace=True)
    smiling_df.drop(["gender"], axis=1, inplace=True)
    smiling_train_df, smiling_test_df = train_test_split(
        smiling_df, test_size=0.2, random_state=66
    )
    return smiling_train_df, smiling_test_df




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
                        shape = predictor(img, d)
                        xmouthpoints = [shape.part(x).x for x in range(48, 67)]
                        ymouthpoints = [shape.part(x).y for x in range(48, 67)]
                        maxx = max(xmouthpoints)
                        minx = min(xmouthpoints)
                        maxy = max(ymouthpoints)
                        miny = min(ymouthpoints)
                        pad = 10

                        crop_image = img[
                            miny - pad : maxy + pad, minx - pad : maxx + pad
                        ]
                        cv2.imwrite(
                            os.path.join(current_smiling_output_path, im["img_name"]),
                            crop_image,
                        )
                else:
                    smiling_df_mouthless.loc[index, "mouthless"] = True
                pbar.update(1)
    # Isolate images which have no detectable region of interest, proceed with 'valid' images
    valid = smiling_df_mouthless[smiling_df_mouthless.mouthless != True]
    invalid = smiling_df_mouthless[smiling_df_mouthless.mouthless == True]
    # Determine proportion of valid images
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

