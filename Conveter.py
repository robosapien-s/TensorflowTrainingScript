import cv2
import pathlib
import os
import tensorflow as tf
import numpy
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader
from pathlib import Path
import time
import random
from datetime import datetime
import argparse


# Construct an argument parser
all_args = argparse.ArgumentParser()

# Add arguments to the parser
all_args.add_argument("-exportDatasetPath", "--exportDatasetPath", required=True)
all_args.add_argument("-datasetVideoPath", "--datasetVideoPath", required=True)
all_args.add_argument("-modelExportPath", "--modelExportPath", required=True)
args = vars(all_args.parse_args())


datasetExportPath = args['exportDatasetPath']

def videoToPictures(videopath, exportpath):
  vidcap = cv2.VideoCapture(videopath)
  success,image = vidcap.read()
  count = 0

  datasetExportPath = Path(exportpath).parent

  print(datasetExportPath)

  p = pathlib.Path(exportpath)
  p.mkdir(parents=True, exist_ok=True)
  
  print(exportpath + '\\' + 'frame%d.png' % time.time_ns())

  while success:
    filePath = exportpath + '\\' + 'frame%d.png' % time.time_ns()
    
    cv2.imwrite(filePath, image)     # save frame as PNG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success, exportpath)
    count += 1

def trainModel(modelExportPath, datasetExportPath):
  print("dataset: " + datasetExportPath)
  print(datasetExportPath)

  print(os.path.exists(datasetExportPath))

  # Load input data specific to an on-device ML app.
  data = DataLoader.from_folder(datasetExportPath)
  train_data, test_data = data.split(0.9)

  # Customize the TensorFlow model.
  model = image_classifier.create(train_data, epochs=100)

  # Evaluate the model.
  loss, accuracy, = model.evaluate(test_data)

  # Export to Tensorflow Lite model and label file in `export_dir`.
  model.export(export_dir=modelExportPath)

datasetVideoPath = args['datasetVideoPath']


for file in pathlib.Path(datasetVideoPath).iterdir():
  for file2 in file.iterdir():
    videoToPictures(os.path.abspath(file2), datasetExportPath + file.name)


modelExportPath = args['modelExportPath']

trainModel(modelExportPath,datasetExportPath)
