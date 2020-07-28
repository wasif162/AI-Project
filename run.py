import sys
from imgmain import *

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "C:\\Users\\Naveed\\Desktop\\Summer 2020\\AI\\Project AI Phase 3\\i1.png"

test = HIP(path=path)
test.imshow("original_image")
print("the path to the image is {}".format(test.get_path()))
print("the width of image is: {}".format(test.width()))
print("the height of image is: {}".format(test.height()))
print("the number of channels of image is: {}".format(test.channel()))

test.save_as("./lena_copy.jpg")

image2 = test.get_image()
cv2.imshow("duplicate", image2)

gray_image = test.rgb_to_gray()
test.imshow("gray")

test.binarize()
test.imshow("binary")

test.reset()
test.resize(new_size=(100,100))
test.imshow("resize by 100,100 ")

test.reset()
test.resize(ratio=(.25,.25), use_ratio=True)
test.imshow("resized by ratio")

test.reset()
test.crop(crop_size=[10, 10, 40, 40])
test.imshow("cropped")

test.reset()
test.flip(mode=1)
test.imshow("flipped")

test.reset()
test.rotate(degree=45, rotate_center=[-50, 50])
test.imshow("rotated by 45")

test.reset()
test.zero_padd(padd_size=[0,10.5,0,10])
test.imshow("zero padded")

test.reset()
test.add_noise()
test.imshow("noise is added")

test.reset()
test.imshow("reset")
dx, dy = test.gradient()
cv2.imshow("dx", dx)
cv2.imshow("dy", dy)

mag = test.magnitude()
cv2.imshow("mag", mag)

edge_im = test.edge()
cv2.imshow("edge", edge_im)

cv2.waitKey(0)