import tensorflow as tf
import tensorflow.keras.backend as K
import math
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import sklearn.metrics
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import tflearn
#from __future__ import print_function, absolute_import, division
#from collections import namedtuple
tf.enable_eager_execution()
#sess = tf.Session()
#sess.run(tf.local_variables_initializer())

def gen():
   #takes the paths of the images and label images, encodes it and zip the appropriate files together  
    
  #path = 'C:/Users/s141010/Documents/school/master/CNN'
  #path = "D:/s141533/NeuralNetworks"
  images_path=glob.glob('C:/Users/s141010/Documents/school/master/CNN/leftImg8bit/test/test/*')#glob.glob(path+"/leftImg8bit/train/aachen/*")
  images_path=[x.encode('utf-8') for x in images_path]  
  labels_path=glob.glob('C:/Users/s141010/Documents/school/master/CNN/leftImg8bit/test/test_label/*')#glob.glob(path+"/gtFine_trainvaltest/gtFine/train/aachen/aachen_[]_gtFine_labellds.png")
  labels_path=[x.encode('utf-8') for x in labels_path]
  couples=zip(images_path,labels_path)
  for paths in couples:
      yield paths   

def add_snow(image):
    image_HLS=tf.image.rgb_to_hsv(image)
    image_HLS =tf.cast(image_HLS,tf.float64)
    brightness_coefficient = 2.5     
    snow_point=140 ## increase this for more snow    
    image_HLS=tf.where(image_HLS<snow_point, )
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)    
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255    
    image_HLS = np.array(image_HLS, dtype = np.uint8)   
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    return image_RGB

def augmentation(image_tensor):
    #nacht
    #image_tensor = tf.image.adjust_saturation(image_tensor, 0.3)
    #image_tensor = tf.image.adjust_brightness(image_tensor, -0.28) 1
    #image_tensor = tf.image.adjust_contrast(image_tensor, 0.9)
    #schemer
    #image_tensor = tf.image.adjust_saturation(image_tensor, 0.6)
    #image_tensor = tf.image.adjust_brightness(image_tensor, -0.15) 
    #image_tensor = tf.image.adjust_contrast(image_tensor, 0.8)
    #image_tensor=add_snow(image_tensor)
    #image_tensor=tflearn.ImageAugmentation()
    #image_tensor.add_random_blur()
    image_tensor=add_snow(image_tensor)
    return image_tensor

def read_image_and_label(image_path,label_path):
  #the mapping function, first the paths will be read and the images saved as tensors. Then the tensors will be normalized in [0,1]
  image_tensor=tf.image.decode_image(tf.io.read_file(image_path))
  image_tensor=tf.math.divide(image_tensor,255) #normalize
  #image_tensor = 2*image_tensor-1
  image_tensor = tf.cast(image_tensor, tf.float32)
  image_tensor=tf.image.resize_image_with_crop_or_pad(image_tensor,256,512)
  #label_tensor=tf.image.decode_image(tf.io.read_file(label_path))
  #label_tensor=tf.image.resize_image_with_crop_or_pad(label_tensor,256,512)
  #label_tensor = tf.cast(label_tensor, tf.int32)
  #new_cids =[0,0,0,0,0,0,0,1,2,0,0,3,4,5,0,0,0,6,0,7,8,9,10,11,12,13,14,15,16,0,0,17,18,19,0]
  #label_new =tf.gather(new_cids, label_tensor)
  aug_image=augmentation(image_tensor)
  return aug_image,image_tensor

def input_fn(batch_size_):
  #make the dataset usable
  buffer_size_ = 10
  dataset = tf.data.Dataset.from_generator(gen,(tf.string,tf.string))
  dataset = dataset.shuffle(buffer_size=buffer_size_,reshuffle_each_iteration=True).repeat(count=None)
  #.repeat(count=None)
  dataset = dataset.map(read_image_and_label)
  #dataset = dataset.map(augmentation)
  dataset = dataset.batch(batch_size_)
  dataset = dataset.prefetch(None)
  return dataset

def test_data(batch_size_):
  #test if the data is what is expected
  dataset=input_fn(batch_size_)
  it=dataset.make_one_shot_iterator()
  next_element=it.get_next()
  for i in range(batch_size_):
    image,label=next_element
    image=image[i,:,:,:]*255
    image=tf.cast(image, tf.int32)
    label=label[i,:,:,:]*255
    label=tf.cast(label, tf.int32)
    #palette= [[0,0,0], [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,  0],[107,142, 35],[152,251,152],[ 70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],[  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32]]
    #label=tf.gather(palette, label)

  plt.imshow(image)
  plt.show()
  plt.imshow(label)
  plt.show()

def sparse_Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

# =============================================================================
# def Mean_IOU(y_true, y_pred):
#     nb_classes = K.int_shape(y_pred)[-1]
#     iou = []
#     true_pixels = K.argmax(y_true, axis=-1)
#     pred_pixels = K.argmax(y_pred, axis=-1)
#     void_labels = K.equal(K.sum(y_true, axis=-1), 0)
#     for i in range(0, nb_classes): # exclude first label (background) and last label (void)
#         true_labels = K.equal(true_pixels, i) & ~void_labels
#         pred_labels = K.equal(pred_pixels, i) & ~void_labels
#         inter = tf.to_int32(true_labels & pred_labels)
#         union = tf.to_int32(true_labels | pred_labels)
#         legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
#         ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
#         iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
#     iou = tf.stack(iou)
#     legal_labels = ~tf.debugging.is_nan(iou)
#     iou = tf.gather(iou, indices=tf.where(legal_labels))
#     return K.mean(iou)
# =============================================================================

def TheModel(batch_size_):
  #build a model and get the loss and accuracy
  dataset=input_fn(batch_size_)
  model=tf.keras.models.Sequential()
  inputs=tf.keras.layers.Input(shape=(256,512,3))
  model.add(ResNet50(include_top=False, weights='imagenet',input_tensor=inputs, pooling=None, classes = 20))
  #kernel_constraint = tf.keras.constraints.max_norm(1.),activation = 'relu',kernel_regularizer = k.regularizers.l2(l=0.1)
  model.add(tf.keras.layers.Conv2D(20,(3,3)))
  #model.add(tf.keras.layers.Dense(units = 20))
  model.add(layers.BatchNormalization())   
  model.add(layers.Dropout(rate=0.7))
  model.add(tf.keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x,(256,512))))
  #model.add(tf.keras.layers.Softmax(axis=-1))
  model.summary()
  sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.8, nesterov=False)
  model.compile(optimizer=sgd,   #SGD with momemtum usually gives good enough results without too many parameters as is the case for ADAMoptimizer
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True), #https://gombru.github.io/2018/05/23/cross_entropy_loss/ This site gives good info about how it works it is basically a sofmax function on the groundtruth scores for the classes followed by the cross-entropy loss on the result. The CE simply calculates a score for a ground truths of all classes compared with the log of s ground truth. CE = -log ( \frac{e^{s_{p}}}{\sum_{j}^{C} e^{s_{j}}} 
                metrics=['accuracy',sparse_Mean_IOU]) #accuracy = (tp + tn) / (tp + fp + fn + tn), precision = TP / (TP + FP), recall = TP / (TP + FN) with TP = tf.count_nonzero(predicted * actual) TN = tf.count_nonzero((predicted - 1) * (actual - 1)) FP = tf.count_nonzero(predicted * (actual - 1)) FN = tf.count_nonzero((predicted - 1) * actual), rmse : root of squared differences between scores and ground truths  
  history=model.fit(dataset, epochs=5, steps_per_epoch=43)

# =============================================================================
#   for layer in model.layers:
#     weights = layer.get_weights() # list of numpy arrays
#     #print(len(weights),tf.reduce_mean(weights),tf.reduce_max(weights),tf.reduce_min(weights))
#   print("model inputs")
#   print(model.input,inputs)
#   print("model outputs")
#   print(model.output)
# =============================================================================

  plt.plot(history.history['acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.show()
  
  plt.plot(history.history['loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.show()
 
    
  plt.plot(history.history['sparse_Mean_IOU'])
  plt.title('mean intersection over union')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.show()
  return model

def test_model(batch_size_):
  #test if the output is what is expected; prints model generated images of the last batch
  dataset=input_fn(batch_size_)
  model=TheModel(batch_size_)
  #output=model.predict(dataset,steps=1)
  #print(output)
  it=dataset.make_one_shot_iterator()
  next_element=it.get_next()
  for i in range(batch_size_):
    image_original,label_original=next_element
    output = model.predict_on_batch(image_original)
    pred = output[i,:,:,:]
    print("output")
    print(pred)
    pred=tf.keras.backend.argmax(pred,axis=-1)
    print("post-argmax predictions")
    print(pred) 
    palette= [[0,0,0], [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,  0],[107,142, 35],[152,251,152],[ 70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],[  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32]]
    pred=tf.gather(palette, pred)    
    image_original = image_original[i,:,:,:]*255
    #image_original=(image_original[i,:,:,:]+1)*127.5
    image_original=tf.cast(image_original, tf.int32)
    label_original=label_original[i,:,:,:]
    label_original=tf.gather(palette, label_original)
    label_original=tf.squeeze(label_original)
    plt.imshow(pred)
    plt.show()
    plt.imshow(image_original)
    plt.show()
    plt.imshow(label_original)
    plt.show()
 
if __name__=='__main__':
  batch_size_ = 4
  test_data(batch_size_)
  #test_model(batch_size_)
  