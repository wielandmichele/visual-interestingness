import tensorflow as tf
from typing import List, Tuple, Dict, Generator

def process_img(filename: str, img_height = 150, img_width = 150) -> tf.Tensor: 
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_height, img_width))
    return img

def process_img_label(filename: str, label: int, img_height = 150, img_width = 150) -> Tuple[tf.Tensor, int]:
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_height, img_width))
    return img, label

def data_generator(inputs: List[str], labels_1: List[int], labels_2: List[int]) -> Generator[Tuple[str, Dict[str, int]], None, None]:
    for i in range(len(inputs)):
        yield inputs[i], {'target_branch': labels_1[i], 'protected_branch': labels_2[i]}