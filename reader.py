import tensorflow as tf
import matplotlib.pyplot as plt
import json
import sys
import os

def dump_json_to_file(input, output):
  with open(input, 'r') as f:
    load_dict = json.load(f)
    js = json.dumps(load_dict, sort_keys=True, indent=2, separators=(',', ':'))
    with open(output, 'w') as dump_f:
      dump_f.write(js)

def load_images(example_path):
  file_path_list = []
  image_name_list = []
  for parent, dirnames, filenames in os.walk(example_path):
    for filename in filenames:
      file_path_list.append(os.path.join(parent, filename))
      image_name_list.append(filename)
  images = load_and_preprocess_image(file_path_list[0:5], 100, 100)
  return (images,image_name_list[0:5])

def load_and_preprocess_image(files, height, width):
  img_tensor_list = []
  img_name_list= []
  for img in files:
    img_raw = tf.io.read_file(img)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    img_final = tf.image.resize(img_tensor, [height, width])
    img_final /= 255.0
    img_tensor_list.append(img_final)
  return img_tensor_list

def load_labels(file, image_names):
  with open(file, 'r') as instance_file:
    load_dict = json.load(instance_file)
    images = load_dict["images"]
    annotations = load_dict['annotations']
    categories = load_dict['categories']
    category_id_list = []
    for image_name in image_names:
      image_id = get_image_id(image_name, images)
      image_category_id = get_image_category(image_id, annotations)
      category_id_list.append(image_category_id)
    return (category_id_list, categories)

def get_image_id(image_name, images):
  for image in images:
    if image_name == image["file_name"]:
      return image["id"]

def get_image_category(image_id, annotations):
  for annotation in annotations:
    if image_id == annotation["image_id"]:
      return annotation["category_id"]
  return 0 # unknown

def show_image_for_test(image_tensor, category):
  fig = plt.figure()
  image_count = len(image_tensor)
  image_row = (image_count-1) / 2 + 1
  for i in range(image_count):
    fig.add_subplot(image_row,2,i+1)
    plt.imshow(image_tensor[i])
    plt.ylabel(category[i])
  plt.show()

def load_data(image_path, annotation_path):
  with tf.Session() as sess:
    images,name_list = load_images(image_path)
    images = sess.run(images)
    labels,categories = load_labels(annotation_path, name_list)
    return images,labels,categories

def load_data_test():
  samples,labels,categories = load_data("../data_set/train2014/", "../data_set/annotations/instances_train2014.json")
  categories_name = []
  for label in labels:
    for category in categories:
      if label == category["id"]:
        categories_name.append(category["name"])
  show_image_for_test(samples, categories_name)

if __name__ == "__main__":
  load_data_test()