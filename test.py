from Model import Model
import Process
model = Model(3, (1,2,3), n_conv = 10)

train = Process.get_image_data('flowers.jpg', size = (368,368))
train = Process.normalize(train, mode = 'forward')
model.train([1,2,3], train, n_iterations = 300)
i = model.generate([1,2,3])
i = Process.normalize(i, mode = 'back')
Process.write_image_data(i, 'second_trained_2.jpg')
