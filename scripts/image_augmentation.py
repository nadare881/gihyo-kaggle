import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from math import pi

# 画像拡張コード
# func(image, M)で変換し、Mは[0,10]で強さを調節する。


# アフィン変換
## 画像を切り抜き、拡大する
@tf.function(experimental_relax_shapes=True)
def crop_and_resize(image, M):
    scale = tf.random.uniform([1], minval=1.-0.5/10*tf.cast(M, "float32"), maxval=1.)
    original_shape = tf.cast(tf.slice(tf.shape(image), [0], [2]), "float32")
    crop_shape = tf.concat([tf.cast(original_shape * scale, "int32"), [3]], axis=0)
    image = tf.image.random_crop(image, crop_shape)
    image = tf.image.resize(image, tf.cast(original_shape, "int32"))
    return image

## 画像を縮小し、端を埋める
@tf.function(experimental_relax_shapes=True)
def shrink_and_pad(image, M):
    scale = tf.random.uniform([1], minval=1.-0.5/10*tf.cast(M, "float32"), maxval=1.)
    original_shape = tf.cast(tf.slice(tf.shape(image), [0], [2]), "float32")
    shrink_shape = tf.cast(original_shape * scale, "int32")
    shape_delta = original_shape - tf.cast(shrink_shape, "float32") + 1.
    top_left = tf.cast(tf.random.uniform([2], minval=[0., 0.], maxval=shape_delta, dtype="float32"), "int32")
    original_shape = tf.cast(original_shape, "int32")
    image = tf.image.resize(image, shrink_shape)
    image = tf.image.pad_to_bounding_box(image, top_left[0], top_left[1], original_shape[0], original_shape[1])
    return image

## 画像を回転させる
@tf.function(experimental_relax_shapes=True)
def rotate(image, M):
    angle = pi / 4 * tf.cast(M, "float32") / 10 * tf.random.uniform([1], minval=-1., maxval=1.)
    image = tfa.image.rotate(image, angle)
    return image

## 画像をx方向にせん断する
@tf.function(experimental_relax_shapes=True)
def shear_x(image, M):
    level = 0.6 * tf.cast(M, "float32") / 10 * tf.random.uniform([1], minval=-1., maxval=1.)
    original_dtype = image.dtype
    image = tf.cast(tfa.image.shear_x(tf.cast(image, "uint8"), level[0], [0, 0, 0]), original_dtype)
    return image

## 画像をy方向にせん断する
@tf.function(experimental_relax_shapes=True)
def shear_y(image, M):
    level = 0.6 * tf.cast(M, "float32") / 10 * tf.random.uniform([1], minval=-1., maxval=1.)
    original_dtype = image.dtype
    image = tf.cast(tfa.image.shear_y(tf.cast(image, "uint8"), level[0], [0, 0, 0]), original_dtype)
    return image

## 画像を上下左右に移動させる
@tf.function(experimental_relax_shapes=True)
def translate_xy(image, M):
    scale = 0.5 * tf.cast(M, "float32") / 10. * tf.random.uniform([2], minval=-1, maxval=1.)
    original_shape = tf.cast(tf.slice(tf.shape(image), [0], [2]), "float32")
    delta = tf.cast(scale * original_shape, "float32")
    original_dtype = image.dtype
    image = tf.cast(tfa.image.translate(tf.cast(image, "uint8"), delta), original_dtype)
    return image

## 画像の縦横比を変える
@tf.function(experimental_relax_shapes=True)
def change_aspect(image, M):
    scale = tf.random.uniform([1], minval=1.-0.5/10*tf.cast(M, "float32"), maxval=1.)
    original_shape = tf.cast(tf.slice(tf.shape(image), [0], [2]), "float32")
    if tf.random.uniform([1], minval=0, maxval=1.)[0] < .5:
        new_shape = tf.cast([original_shape[0], original_shape[1]*scale[0]], "int32")
    else:
        new_shape = tf.cast([original_shape[0]*scale[0], original_shape[1]], "int32")
    original_shape = tf.cast(original_shape, "int32")
    image = tf.image.resize(image, new_shape)
    image = tf.image.resize_with_pad(image, original_shape[0], original_shape[1])
    return image


# 色空間の変換
## 画像のコントラストを調整する
@tf.function(experimental_relax_shapes=True)
def auto_contrast(image, M):
    original_dtype = image.dtype
    min_q = tf.cast(M, "float32") * 5. * tf.random.uniform([1], minval=0, maxval=1.)[0]
    max_q = 100. - min_q
    new_images = []
    for i in range(3):
        image_slice = tf.cast(tf.slice(image, [0, 0, i], [-1, -1, 1]), "float32")
        x = tf.reshape(image_slice, [-1])
        min_x = tfp.stats.percentile(x, min_q)
        max_x = tfp.stats.percentile(x, max_q)
        scale = 255 / tf.maximum((max_x - min_x), tf.keras.backend.epsilon())
        new_images.append((image_slice - min_x) * scale)
    image = tf.concat(new_images, axis=2)
    return image

## 画像のコントラストを変更する
@tf.function(experimental_relax_shapes=True)
def contrast(image, M):
    factor = 1 + .09 * tf.cast(M, "float32") * tf.random.uniform([1], minval=-1., maxval=1.)[0]
    original_dtype = image.dtype
    image = tf.cast(image, "float32")
    image = tf.clip_by_value((image - 128) * factor + 128, 0., 255.)
    image = tf.cast(image, original_dtype)
    return image

## 画像の輝度を調整する
@tf.function(experimental_relax_shapes=True)
def brightness(image, M):
    factor = 1 + .09 * tf.cast(M, "float32") * tf.random.uniform([1], minval=-1., maxval=1.)[0]
    original_dtype = image.dtype
    image = tf.cast(image, "float32")
    image = tf.clip_by_value(image * factor, 0., 255.)
    image = tf.cast(image, original_dtype)
    return image

## 画像のbit数を減らす
@tf.function(experimental_relax_shapes=True)
def posterize(image, M):
    ibits = tf.cast(M/2*tf.random.uniform([1], minval=0, maxval=1.)[0], "int32")
    b = tf.cast(tf.pow(2, ibits), "uint8")
    original_dtype = image.dtype
    image = tf.cast(image, "uint8") // b * b
    image = tf.cast(image, original_dtype)
    return image


# フィルターの適用
## 平均値をとるフィルターをかける
@tf.function(experimental_relax_shapes=True)
def mean_blur(image, M):
    image = tfa.image.mean_filter2d(image)
    return image

## 中央値をとるフィルターをかける
@tf.function(experimental_relax_shapes=True)
def median_blur(image, M):
    image = tfa.image.median_filter2d(image)
    return image

##画像のsharpnessを調整する
@tf.function(experimental_relax_shapes=True)
def sharpness(image, M):
    factor = 1 + .09 * tf.cast(M, "float32") * tf.random.uniform([1], minval=-1., maxval=1.)[0]
    original_dtype = image.dtype
    image = tf.cast(image, "float32")
    image = tfa.image.sharpness(image, factor)
    image = tf.cast(image, original_dtype)
    return image    


# その他
## 何も変換しない
@tf.function(experimental_relax_shapes=True)
def identity(image, M):
    return image

## 画像の一部を切り抜き0にする
@tf.function(experimental_relax_shapes=True)
def cutout(image, M):
    original_shape = tf.cast(tf.slice(tf.shape(image), [0], [2]), "float32")
    mask_shape = tf.cast(original_shape * tf.cast(M, "float32") / 10 / 4 * tf.random.uniform([2], minval=0., maxval=1., dtype="float32"), "int32") * 2
    image = tfa.image.random_cutout(tf.expand_dims(image, axis=0), mask_shape)[0]
    return image