import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL
import numpy as np
import pytorch as torch
import pickle
import os
import random
from sklearn.preprocessing import OneHotEncoder
from scipy import misc
from functools import reduce
import math
import cv2
from skimage import transform

current_iteration = 0

# change list of labels to one hot encoder
# e.g. [0,1,2] --> [[1,0,0],[0,1,0],[0,0,1]]
def OHE_labels(Y_tr, N_classes):
    OHC = OneHotEncoder()
    Y_ohc = OHC.fit(np.arange(N_classes).reshape(-1, 1))
    Y_labels = Y_ohc.transform(Y_tr.reshape(-1, 1)).toarray()
    return Y_labels

# apply histogram equalization to remove the effect of brightness, use openCV'2 cv2
# scale images between -.5 and .5, by dividing by 255. and subtracting .5.
# this function can be merged with preprocess_image(img, image_size)
def pre_process_image(image):
    # todo Zhanganlan
    # error occurs when cifar-10 is te
    # solve in 2018.5.11 ZhangAnlan
    # the data type of image that used in equalizeHist must be uint8
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    # image = image/255. - .5
    image = image/255.
    #image = image / 127.5 - 1
    return image


def resize_image_with_crop_or_pad(image, target_height, target_width):
	x,y = image.shape[0],image.shape[1]
	if x < target_height or y < target_height:
		w = max(0, (target_width - x) / 2)
		h = max(0, (target_height - y) / 2)
		pad = torchvision.transforms.Pad(w, h, w, h)
		image = transform(image)
	CenterCrop = torchvision.transforms.CenterCrop(int(image.shape[0]), int(image.shape[1]))
	image = CenterCrop(image)
	return image


def randomly_overlay(image, patch, if_random=False):
    # randomly overlay the image with patch
    patch_mask = np.ones([patch.shape[0],patch.shape[0],3], dtype=np.float32)
    patch_mask = torch.Tensor(patch_mask)
    #patch_mask = tf.convert_to_tensor(patch_mask)
    patch_size = int(patch.shape[0])*1.5
	
	patch = resize_image_with_crop_or_pad(patch, int(patch_size), int(patch_size))
	patch_mask = resize_image_with_crop_or_pad(patch, int(patch_size), int(patch_size))
	'''
	??似乎resize后必然比原patch大
    patch = tf.image.resize_image_with_crop_or_pad(patch, int(patch_size), int(patch_size))
    patch_mask = tf.image.resize_image_with_crop_or_pad(patch_mask, int(patch_size), int(patch_size))
	'''

    if if_random==True:
        angle = np.random.uniform(low=-180.0, high=180.0)
        location_x = int(np.random.uniform(low=0, high=int(image.shape[0])-patch_size))
        location_y = int(np.random.uniform(low=0, high=int(image.shape[0])-patch_size))
    else:
        angle = 0
        location_x = 52
        location_y = 48
    '''
    # rotate the patch and mask with the same angle
    def random_rotate_image_func(image, angle):
        return misc.imrotate(image, angle, 'bicubic')
    patch_rotate = tf.py_func(random_rotate_image_func, [patch, angle], tf.uint8)
    patch_mask = tf.py_func(random_rotate_image_func, [patch_mask, angle], tf.uint8)
    patch_rotate = tf.image.convert_image_dtype(patch_rotate, tf.float32)
    patch_mask = tf.image.convert_image_dtype(patch_mask, tf.float32)
    '''
    patch_rotate = patch
    # move the patch and mask to the sama location
	patch_rotate = pad_to_bounding_box(patch_rotate, location_y, location_x, int(image.shape[0]), int(image.shape[0]))
	patch_mask = pad_to_bounding_box(patch_mask, location_y, location_x, int(image.shape[0]), int(image.shape[0]))
	'''
    patch_rotate = tf.image.pad_to_bounding_box(patch_rotate, location_y, location_x, int(image.shape[0]), int(image.shape[0]))
    patch_mask = tf.image.pad_to_bounding_box(patch_mask, location_y, location_x, int(image.shape[0]), int(image.shape[0]))
	'''

    # overlay the image with patch
    image_with_patch = (1-patch_mask)*image + patch_rotate
    return image_with_patch


def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
	transform = torchvision.transforms.Pad(offset_width, offset_height, 0, 0)#left,top,right,bottom
	image = transform(image)
	if target_height > image[2]:
		transform = torchvision.transforms.Pad(0, 0, target_height - image[2], 0)
		image = transform(image)
	if target_width > image[1]:
		transform = torchvision.transforms.Pad(0, 0, 0, target_width - image[1])
		image = transform(image)
	return image
		
# ZhangAnlan 2018.5.3
# param@num number of image/patch to load
# param@data_dir directory of image/patch
# returnVal@ return a pair of list of image/patch and corresponding labels i.e return image, label
# extra=True --> need to generate extra data, otherwise only preprocess
# N_classes, n_each=, ang_range, shear_range, trans_range and randomize_Var are parameters needed to generate extra data

def load_image( num, file_path, N_classes, encode='latin1'):
    image = []
    label = []
    patch = []
    with open(file_path, 'rb') as f:
        # cifar-10 need use 'latin1'
        data = pickle.load(f) # if use python2.7 there should be no argument 'encoding'

    # the names of the keys should be unified as 'data', 'labels'
    # todo Zhanganlan
    # to be removed! liuas test!!!!!!!!
    '''
    if str(file_path).endswith("train.p") or str(file_path).endswith("test.p"):
        temp_image = data['features']
    else: # cifar-10 data set needs some pre-process
        temp_image = data['data']
        temp_image = temp_image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    '''
    temp_image = data['data']
    temp_label = OHE_labels(data['labels'], N_classes)

    while(len(label) < num):
        # pick up randomly
        # liuas test!!
        iter = random.randint(0, len(temp_label) - 1)
        #iter = random.randint(0, 1100)
        image.append(temp_image[iter])
        label.append(temp_label[iter])
        # print(data['labels'][iter])

    # further tests are needed for ZhangAnlan
    image = np.array([pre_process_image(image[i]) for i in range(len(image))], dtype=np.float32)

    return image, label


def get_initial_image_patch_pair(image_num, patch_num, validate=False):
    result_img = []

    # all combinations for images and patches
    for i in range(image_num):
        for j in range(patch_num):
            result_img.append([i,j])

    if not validate:
        np.random.shuffle(result_img)

    return result_img

def get_current_pair(batch_size, pair_set, last_iter):
    current_pair = []
    if last_iter + batch_size < len(pair_set):
        current_pair += pair_set[last_iter: last_iter+batch_size]
        last_iter = last_iter + batch_size
    else:
        current_pair += pair_set[last_iter: ]
        current_pair += pair_set[0: batch_size - len(pair_set) + last_iter]
        last_iter = batch_size- len(pair_set)+last_iter

    np.random.shuffle(current_pair)
    #last_iter = last_iter + batch_size
    return current_pair, last_iter


def load_data_in_pair( pair_set, num, image_path, patch_path, image_classes, encode='latin1'):
    image = []
    label = []
    patch = []
    global current_iteration
    with open(image_path, 'rb') as f:
        # cifar-10 need use 'latin1'
        data_img = pickle.load(f) # if use python2.7 there should be no argument 'encoding'

    with open(patch_path, 'rb') as f:
        # cifar-10 need use 'latin1'
        data_pat = pickle.load(f) # if use python2.7 there should be no argument 'encoding'


    temp_image = data_img['data']
    temp_label = OHE_labels(data_img['labels'], image_classes)

    temp_patch = data_pat['data']

    current_pair_set, current_iteration = get_current_pair(num, pair_set, current_iteration)
    #print(current_iteration)

    for i in range(num):
        image.append(temp_image[current_pair_set[i][0]])
        label.append(temp_label[current_pair_set[i][0]])
        patch.append(temp_patch[current_pair_set[i][1]])

    # further tests are needed for ZhangAnlan
    image = np.array([pre_process_image(image[i]) for i in range(len(image))], dtype=np.float32)
    patch = np.array([pre_process_image(patch[i]) for i in range(len(patch))], dtype=np.float32)

    return image, label, patch

# load and augment patch, image with different combinations
def shuffle_augment_and_load(image_num, image_dir, patch_num, patch_dir, batch_size):

    if batch_size <= 0:
        return None

    # load image/patch from directory
    image_set, image_label_set = load_image(image_num, image_dir, N_classes=43)
    patch_set, _ = load_image(patch_num, patch_dir, N_classes=10)

    result_img = []
    result_patch = []
    result_img_label = []

    # all combinations for images and patches
    for i in range(image_num):
        for j in range(patch_num):
            result_img.append(image_set[i])
            #label_tensor = tf.one_hot(image_label_set[i], 43)
            result_img_label.append(image_label_set[i])
            result_patch.append(patch_set[j])

    # more, need to be croped
    if len(result_img) >= batch_size:
        result_img = result_img[:batch_size]
        result_img_label = result_img_label[:batch_size]
        result_patch = result_patch[:batch_size]
        return result_img,result_img_label,result_patch

    # not enough, random pick
    else:
        len_to_supp = batch_size - len(result_img)
        for iter in range(len_to_supp):
            ran_img = random.randint(0, image_num - 1)
            result_img.append(image_set[ran_img])
            #label_tensor = tf.one_hot(image_label_set[ran_img], 43)
            result_img_label.append(image_label_set[ran_img])
            result_patch.append(patch_set[random.randint(0, patch_num - 1)])
        return result_img,result_img_label,result_patch

# TV, distance between a pixel and its adjacent 2 pixels.
# In order to make patch more 'smooth'
# warning: be ware of 'inf'
# not TESTED!
def TV(patch, patch_size, batch_size):
    if not batch_size == patch.shape[0]:
        return None

    # TV for single image
	'''
    def single_image_TV(patch, patch_size):
        result = tf.Variable(tf.zeros([1, patch_size - 1, 3]))
        slice_result = tf.assign(result, patch[0: 1, 1:, 0: 3])
        for iter in range(1, patch_size - 1):
            temp = tf.assign(result,tf.add(tf.subtract(patch[iter:iter + 1, 1:, 0: 3], patch[iter:iter + 1, 0:-1, 0: 3]),
                                           tf.subtract(patch[iter:iter + 1, 0:-1, 0: 3],patch[iter + 1:iter + 2, 0:-1, 0: 3])))
            slice_result = tf.concat([slice_result, temp], 0)

            return slice_result
	'''
	#assuming patch is [size*size*3] here
	#unchecked
	def single_image_TV(patch, patch_size):
		result = numpy.zeros((1,patch_size - 1, 3))
		result[1] = [x for x in range(1,patch_size)]
		result[2] = [x for x in range(0,3)]
		slice_result = torch.from_numpy(result)
		for iter in range(1, patch_size - 1):
				sub1 = patch[iter:iter + 1, 1:, 0: 3] + patch[iter:iter + 1, 0:-1, 0: 3]
				sub2 = patch[iter:iter + 1, 0:-1, 0: 3] + patch[iter + 1:iter + 2, 0:-1, 0: 3]
				temp = sub1 + sub2
				slice_result = torch.cat([slice_result, temp],0)
				return slice_result

    batch_image = patch[0]
    batch_image = single_image_TV(batch_image, patch_size)
    batch_image = torch.unqueeze(batch_image, 0)
	#batch_image = tf.expand_dims(batch_image, 0)
    for iter in range(1, batch_size):
        temp = single_image_TV(patch[iter], patch_size)
        temp = torch.unsqueeze(temp, 0)
		#temp = tf.expand_dims(temp, 0)
        batch_image = torch.cat([batch_image, temp], 0)
		#batch_image = tf.concat([batch_image, temp], 0)
		
    return l2_loss(batch_image)
	#return torch.nn.l2_loss(batch_image)

def l2_loss(img):
	image_numpy = img.numpy()
	image_l2 = np.square(image_numpy)
	loss = np.sum(image_l2, axis=1)
	loss = np.sum(l2_loss, axis=0)
	return loss

# total variance loss
def tv_loss(image, tv_weight):
    # total variation denoising
    shape = tuple(image.get_shape().as_list())
    tv_y_size = _tensor_size(image[:,1:,:,:])
    tv_x_size = _tensor_size(image[:,:,1:,:])
    tv_loss = tv_weight * 2 * ((l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) / tv_y_size) +
            (l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) / tv_x_size))

    return tv_loss
	
#unchecked!!
def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

# save tensor
def save_obj(tensor, filename):
    tensor = np.asarray(tensor).astype(np.float32)
    # print(b.eval())
    serialized = pickle.dumps(tensor, protocol=0)
    with open(filename, 'wb') as f:
        f.write(serialized)

# save patches
def save_patches(patches, filename):
    num = int(math.sqrt(int(patches.shape[0])))
    for i in range(num):
        for j in range(num):
            temp = resize_image_with_crop_or_pad(patches[i*num+j], int(patches.shape[1])+2, int(patches.shape[1])+2)
            if not j:
                row = temp
            else:
                row = torch.cat([row, temp], 1)
        if not i:
            show_patch = row
        else:
            show_patch = torch.cat([show_patch, row], 0)
        del row
    plt.figure(figsize=(5,5))
    plt.imshow(_convert(show_patch.eval()))
    plt.axis('off')
    plt.savefig(filename, dpi=200)
    plt.close()


# load tensor
def load_obj(filename):
    if os.path.exists(filename):
        return None
    with open(filename, 'rb') as f:
        tensor = pickle.load(f)
    tensor = np.asarray(tensor).astype(np.float32)
    return tensor

def _convert(image):
    return (image * 255.0).astype(np.uint8)
    #return ((image + 1.0) * 127.5).astype(np.uint8)

# show image
def show_image(image):
    plt.axis('off')
    plt.imshow(_convert(image), interpolation="nearest")
    plt.show()

# plot accrucy
def plot_acc(acc, filename):
    plt.plot(acc)
    plt.ylabel('Accrucy')
    plt.savefig(filename, dpi=200)
    plt.close()

# show image with patch and accuracy
def plot_images_and_acc(image, result, acc, num, filename):
    size = int(math.ceil(math.sqrt(num)))
    fig = plt.figure(figsize=(5,5))
    fig.suptitle('Accuracy of misclassification: %4.4f' % acc, verticalalignment='top')
    for i in range(size):
        for j in range(size):
            if(i*size+j < num):
                temp = image[i*size+j]
                p = fig.add_subplot(size,size,i*size+j+1)
                # p = plt.subplot(size,size,i*size+j)
                p.imshow(_convert(temp.eval()))
                p.axis('off')
                if(result[i*size+j]!=0):
                    p.set_title("Wrong", fontsize=8)
                else:
                    p.set_title("Right", fontsize=8)
    # plt.title('Accuracy of misclassification: %4.4f' % acc)
    fig.savefig(filename, dpi=200)
