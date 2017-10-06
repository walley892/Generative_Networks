from Model import Model
import Process
from numpy import random as rd
import numpy as np

n_params = 10

model = Model(n_params, (1,2,3), dense_activation = 'relu',n_upsample = 5, n_conv = 10, conv_kernels = [8, 16,32,64, 32, 32, 16, 16, 8, 8])

model.load('model.h5')
data, labels = Process.get_data_for_abstract('./abs_1', size = (736,736), n_params = n_params)
print(labels)

data = Process.normalize(data)
for var in range(100):
    print(var)
    model.train(labels, data, n_iterations = 1)
    if var % 50 == 0:
        i = model.generate(rd.rand(10,n_params)[0])
        i = Process.normalize(i,mode = 'back')
        Process.write_image_data(i[0], 'progress.jpg')

model.save('model.h5')

for variation in range(10):
    i = model.generate(rd.rand(10,n_params)[0])
    i = Process.normalize(i, mode = 'back')
    Process.write_image_data(i[0], 'param_variations/variation_' + str(variation)+ '.jpg')

for j in range(20):
    i = model.generate([j/19 for _ in range(n_params)])
    i = Process.normalize(i, mode = 'back')
    Process.write_image_data(i[0], 'param_gradient/grad_' + str(j) + '.jpg')
def move_toward(a,axis, amt):
    a[axis] += amt
    a[0:axis] -= amt/len(a)
    a[axis+1:] -= amt/len(a)
    return a

