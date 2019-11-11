import sys

sys.path.insert(0, '../sim/')
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def huber(x, y, c=1.0):
    diff = x - y
    l2 = tf.square(diff)
    l1 = tf.abs(diff)
    # c = (ratio)*tf.reduce_max(diff)
    diff = tf.where(tf.greater(diff, c), 0.5 * tf.square(c) + c * (l1 - c), 0.5 * l2)
    return diff

def mean_huber(x, y, mask=None):
    """
    Mean huber loss
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)

    return tf.reduce_mean(huber(x, y) * mask)

def sum_huber(x, y, mask=None):
    """
    Sum huber loss
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)

    return tf.reduce_sum(huber(x, y) * mask)

def ZNCC(x,y):
	"""
	ZNCC dissimilarity measure (Zero Mean Normalized Cross-Correlation)
	Args:
		x: predicted image
		y: target image
	"""
	mean_x = tf.reduce_mean(x)
	mean_y = tf.reduce_mean(y)
	norm_x = x-mean_x
	norm_y = y-mean_y
	variance_x = tf.sqrt(tf.reduce_sum(tf.square(norm_x)))
	variance_y = tf.sqrt(tf.reduce_sum(tf.square(norm_y)))

	zncc = tf.reduce_sum(norm_x*norm_y)/(variance_x*variance_y)
	return 1-zncc

def SSIM(x, y):
    """
    SSIM dissimilarity measure
    Args:
        x: predicted image
        y: target image
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID')
    mu_y = tf.nn.avg_pool(y, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID')

    sigma_x = tf.nn.avg_pool(x ** 2, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID') - mu_x ** 2
    sigma_y = tf.nn.avg_pool(y ** 2, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool(x * y, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def SSIM_l1(x,y,alpha=0.85):
	ss = tf.pad(SSIM(x,y),[[0,0],[1,1],[1,1],[0,0]])
	ll = l1(x,y)
	return alpha*ss+(1-alpha)*ll

def mean_SSIM(x,y):
	"""
	Mean error over SSIM reconstruction
	"""
	return tf.reduce_mean(SSIM(x,y))

def mean_SSIM_l1(x, y):
	return 0.4* mean_SSIM(x, y) + 0.6 * mean_l1(x, y)

def smoothness(x, y):
    """
    Smoothness constraint between predicted and image
    Args:
        x: disparity
        y: image
    """

    def gradient_x(image):
        sobel_x = tf.Variable(initial_value=[[1, 0, -1], [2, 0, -2], [1, 0, -1]], trainable=False, dtype=tf.float32)
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        if image.get_shape()[-1].value == 3:
            sobel_x = tf.concat([sobel_x, sobel_x, sobel_x], axis=2)
        return tf.nn.conv2d(image, sobel_x, [1, 1, 1, 1], padding='SAME')

    def gradient_y(image):
        sobel_y = tf.Variable(initial_value=[[1, 2, -1], [0, 0, 0], [-1, -2, -1]], trainable=False, dtype=tf.float32)
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        if image.get_shape()[-1].value == 3:
            sobel_y = tf.concat([sobel_y, sobel_y, sobel_y], axis=2)
        return tf.nn.conv2d(image, sobel_y, [1, 1, 1, 1], padding='SAME')

    # normalize image and disp in a fixed range
    x = x / 255
    y = y / 255

    disp_gradients_x = gradient_x(x)
    disp_gradients_y = gradient_y(x)

    image_gradients_x = tf.reduce_mean(gradient_x(y), axis=-1, keepdims=True)
    image_gradients_y = tf.reduce_mean(gradient_y(y), axis=-1, keepdims=True)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

    smoothness_x = tf.abs(disp_gradients_x) * weights_x
    smoothness_y = tf.abs(disp_gradients_y) * weights_y

    return tf.reduce_mean(smoothness_x + smoothness_y)

def mean_l1(x, y, mask=None):
    """
    Mean reconstruction error
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)
    return tf.reduce_sum(mask * tf.abs(x - y)) / tf.reduce_sum(mask)

def mean_l2(x, y, mask=None):
    """
    Mean squarred error
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)
    return tf.reduce_sum(mask*tf.square(x - y)) / tf.reduce_sum(mask)

def sum_l1(x, y, mask=None):
    """
    Sum of the reconstruction error
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)
    return tf.reduce_sum(mask * tf.abs(x - y))

def sum_l2(x, y, mask=None):
    """
    Sum squarred error
    Args:
        x: predicted image
        y: target image
        mask: compute only on those points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)
    return tf.reduce_sum(mask * tf.square(x - y))

def sign_and_elementwise(x,y):
	"""
	Return the elementwise and of the sign between vectors
	"""
	element_wise_sign = tf.sigmoid(10*(tf.sign(x)*tf.sign(y)))
	return tf.reduce_mean(tf.sigmoid(element_wise_sign))

def cos_similarity(x,y,normalize=False):
	"""
	Return the cosine similarity between (normalized) vectors
	"""
	if normalize:
		x = tf.nn.l2_normalize(x)
		y = tf.nn.l2_normalize(y)
	return tf.reduce_sum(x*y)

SUPERVISED_LOSS = {
    'mean_l1': mean_l1,
    'sum_l1': sum_l1,
    'mean_l2': mean_l2,
    'sum_l2': sum_l2,
    'smoothness': smoothness,
    'SSIM':SSIM,
	'SSIM_l1':SSIM_l1,
    'mean_SSIM':mean_SSIM,
	'mean_SSIM_l1':mean_SSIM_l1,
	'ZNCC':ZNCC,
	'cos_similarity':cos_similarity,
    'huber':huber,
    'mean_huber':mean_huber,
	'sum_huber':sum_huber
}

ALL_LOSSES = dict(SUPERVISED_LOSS)

def get_supervised_loss(name, x, y, mask=None):
    if name not in ALL_LOSSES.keys():
        print('Unrecognized loss function, pick one among: {}'.format(ALL_LOSSES.keys()))
        raise Exception('Unknown loss function selected')
    base_loss_function = ALL_LOSSES[name]
    return base_loss_function(x, y, mask)