import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test4.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

stop_data = cv2.CascadeClassifier('cascade.xml')

found = stop_data.detectMultiScale(img_gray,
                                   minSize=(50, 50))

amount_found = len(found)

print(amount_found)

for (x, y, width, height) in found:
    # We draw a green rectangle around
    # every recognized sign
    cv2.rectangle(img_rgb, (x, y),
                  (x + width, y + height),
                  (0, 255, 0), 5)
# Creates the environment of
# the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()