import numpy as np
import cv2
import imutils


def disp(img1,img2):
    image = img1
    image2 = img2
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grey = imutils.resize(grey, width=512, height = 512)
    image = imutils.resize(image, width=512,height = 512)
    image2 = imutils.resize(image2, width=512,height = 512)
    grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    final_stack =  np.hstack((image, grey_3_channel, image2))
    cv2.imshow('Output', final_stack)
    cv2.waitKey()
    return final_stack

def save(stack):
    filename = 'Ans/poke5.png'
    cv2.imwrite(filename, stack) 

net = cv2.dnn.readNetFromCaffe("data/colorization_deploy_v2.prototxt", "data/colorization_release_v2.caffemodel")
pts = np.load("data/pts_in_hull.npy")
image = cv2.imread("png/poke5.png")


class8, conv8 = net.getLayerId("class8_ab"), net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

scaled_img = image.astype("float32") / 255.0
lab_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2LAB)
resized_img = cv2.resize(lab_img, (224, 224))
L = cv2.split(resized_img)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab_img)[0]
colorized_img = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized_img = cv2.cvtColor(colorized_img, cv2.COLOR_LAB2BGR)
colorized_img = np.clip(colorized_img, 0, 1)
colorized_img = (255 * colorized_img).astype("uint8")

save(disp(image,colorized_img))
