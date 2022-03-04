from ninja.files import UploadedFile
from ninja import File
import tensorflow as tf
import numpy as np
from nstapp.apps import NstappConfig
from io import BytesIO
from PIL import Image



def upload_tensor_img(bucket, tensor, key):
    tensor = np.array(tensor * 255, dtype=np.uint8)
    image = Image.fromarray(tensor[0])
    buffer = BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    NstappConfig.s3.put_object(Bucket=bucket, Key=key, Body=buffer, ACL='public-read', ContentType='jpeg')
    location = NstappConfig.s3.get_bucket_location(Bucket=bucket)['LocationConstraint']
    url = f"https://s3-{location}.amazonaws.com/{bucket}/{key}"
    return url


def load_style(path_to_style, max_dim):
    img = tf.io.read_file(path_to_style)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def nst_apply(key: str, img: UploadedFile = File(...)) -> str:
    style_path = tf.keras.utils.get_file('kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

    img = Image.open(img.file).convert('RGB')
    content_image = tf.keras.preprocessing.image.img_to_array(img)
    style_image = load_style(style_path, 512)
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    content_image = tf.image.resize(content_image, (256, 256))

    stylized_image = NstappConfig.hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    image_url = upload_tensor_img('nst10', stylized_image, key)
    return image_url


def nst2_apply(key: str, img: UploadedFile = File(...)) -> str:
    style_path = tf.keras.utils.get_file('mosaic5.jpg', 'https://www.erinhanson.com/Content/InventoryImages/Erin-Hanson-Aspen-Mosaic-2.jpg')

    img = Image.open(img.file).convert('RGB')
    content_image = tf.keras.preprocessing.image.img_to_array(img)
    style_image = load_style(style_path, 524)
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    content_image = tf.image.resize(content_image, (256, 256))

    stylized_image = NstappConfig.hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    image_url = upload_tensor_img('nst10', stylized_image, key)
    return image_url


def nst3_apply(key: str, img: UploadedFile = File(...)) -> str:
    style_path = tf.keras.utils.get_file('piccasso5.jpg', 'https://www.pablo-ruiz-picasso.net/images/works/1906.jpg')

    img = Image.open(img.file).convert('RGB')
    content_image = tf.keras.preprocessing.image.img_to_array(img)
    style_image = load_style(style_path, 578)
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    content_image = tf.image.resize(content_image, (256, 256))

    stylized_image = NstappConfig.hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    image_url = upload_tensor_img('nst10', stylized_image, key)
    return image_url


def nst4_apply(key: str, img: UploadedFile = File(...)) -> str:
    style_path = tf.keras.utils.get_file('monet5.jpg', 'https://www.erinhanson.com/Content/InventoryImages/Erin-Hanson-Dappled-Light.jpg')

    img = Image.open(img.file).convert('RGB')
    content_image = tf.keras.preprocessing.image.img_to_array(img)
    style_image = load_style(style_path, 524)
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    content_image = tf.image.resize(content_image, (256, 256))

    stylized_image = NstappConfig.hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    image_url = upload_tensor_img('nst10', stylized_image, key)
    return image_url
