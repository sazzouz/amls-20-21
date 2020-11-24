"""
Data preprocessing operations.
"""
# Import packages
import os
import shutil
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(celeba_df):
  gender_df = celeba_df.copy()
  gender_df.drop(gender_df.columns[0], axis=1, inplace=True)
  gender_df.drop(['smiling'], axis=1, inplace=True)
  gender_train_df, gender_test_df = train_test_split(
    gender_df, 
    test_size=0.2,
    random_state=66
    )
  return gender_train_df, gender_test_df

def process_data(task_dir, celeba_img_dir, gender_df, usage, feature):
  gender_df_faceless = gender_df.copy()
  gender_df_faceless['faceless'] = np.nan
  for i in tqdm(sorted(gender_df.gender.unique()), desc="Mapping Classified Celeba Images To Temp Train/Test Directories"):
      print('Creating Directory For Class {0}'.format(i))
      current_gender_output_path = os.path.join(task_dir, str(usage), str(i))
      if os.path.exists(current_gender_output_path) and os.path.isdir(current_gender_output_path):
          shutil.rmtree(current_gender_output_path)
      try:
          os.makedirs(current_gender_output_path)
      except OSError:
          print ("Failed To Create Directory %s" % current_gender_output_path)
      else:
          print ("Created Directory %s " % current_gender_output_path)
      is_current_gender = gender_df[str(feature)] == str(i)
      current_gender = gender_df[is_current_gender]
      with tqdm(total=current_gender.shape[0]) as pbar:
          pbar.set_description("Processing Images For Class %s" % i)
          for index, im in current_gender.iterrows():
              im_path = os.path.join('..', celeba_img_dir, im['img_name'])
              img = cv2.imread(im_path)
              # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              face_cascade=cv2.CascadeClassifier('haarcascade_face.xml')
              get_face=face_cascade.detectMultiScale(img)
              if list(get_face):
                  locs=[]
                  for (x,y,w,h) in get_face:
                      locs.append(img[y:y+h, x:x+w])
                  cv2.imwrite(os.path.join(current_gender_output_path, im['img_name']), locs[0])
              else:
                  gender_df_faceless.loc[index, 'faceless'] = True
              pbar.update(1)
  is_valid = gender_df_faceless[gender_df_faceless.faceless != True]
  is_not_valid = gender_df_faceless[gender_df_faceless.faceless == True]
  perc_valid = str(is_valid.shape[0]*100/gender_df.shape[0]) + '%'
  print(perc_valid + ' ' + 'of the dataset provided have detectable faces.')
  perc_reduction = str(round(100 - is_valid.shape[0]*100/gender_df.shape[0], 2)) + '%'
  print(perc_reduction + ' ' + 'of the dataset provided will therefore not be processed.')
  return is_valid, is_not_valid
  

def gen_data(task_dir):
  train_imgen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        zca_epsilon=2e-06,
        rotation_range=45,
        width_shift_range=4.0,
        height_shift_range=2.0,
        brightness_range=[0.3, 1.0],
        shear_range=2.0,
        zoom_range=24.2,
        channel_shift_range=3.6,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
        )
  test_imgen = ImageDataGenerator(rescale=1./255)
  gender_train_flow = train_imgen.flow_from_directory(
          os.path.join(task_dir, 'train'),
          color_mode="rgb",
          class_mode="sparse",
          target_size=(218,178),
          shuffle=True,
          seed=66,
          subset="training")
  gender_val_flow = train_imgen.flow_from_directory(
        os.path.join(task_dir, 'train'),
        color_mode="rgb",
        class_mode="sparse",
        target_size=(218,178),
        shuffle = True,
        seed=66,
        subset="validation"
        )
  gender_test_flow = test_imgen.flow_from_directory(
        os.path.join(task_dir, 'test'),
        color_mode="rgb",
        class_mode="sparse",
        target_size=(218,178),
        shuffle = True,
        seed=66)
  return gender_train_flow, gender_val_flow, gender_test_flow