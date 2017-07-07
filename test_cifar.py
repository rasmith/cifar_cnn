import cifar
from matplotlib import pyplot as plt

num_train_images = 10
num_test_images = 10
width = 32
height = 32
channels = 3
c = cifar.Cifar(num_train_images, num_test_images)
((train_images, train_labels), (test_images, test_labels)) = c.load_data()
print('train_labels = %s' %  \
    (','.join([c.labels[i]  for i in c.train_labels])))
plt.imshow(train_images[0])
plt.show()
print('test_labels = %s' %  \
    (','.join([c.labels[i]  for i in c.test_labels])))
plt.imshow(test_images[0])
plt.show()

