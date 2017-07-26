from __future__ import print_function

import numpy as np
import tensorflow as tf

import utils
from src.data import DATA_PATH


def main():
    with open(DATA_PATH + "/vgg16/vgg16.tfmodel", mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float", [None, 224, 224, 3])

    tf.import_graph_def(graph_def, input_map={"images": images})
    print("graph loaded from disk")

    graph = tf.get_default_graph()

    a = [n.name for n in graph.as_graph_def().node]
    for name in a:
        print(name)
    cat = utils.load_image("cat.jpg")
    a = ['import/conv1_1/Relu', 'import/conv1_2/Relu', 'import/pool1', 'import/conv2_1/Relu', 'import/conv2_2/Relu',
         'import/pool2',
         'import/conv3_1/Relu',
         'import/conv3_2/Relu',
         'import/conv3_3/Relu',
         'import/pool3',
         'import/conv4_1/Relu',
         'import/conv4_2/Relu',
         'import/conv4_3/Relu',
         'import/conv5_1/Relu',
         'import/conv5_2/Relu',
         'import/conv5_3/Relu',
         'import/pool5'
         ]

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        print("variables initialized")

        batch = cat.reshape((1, 224, 224, 3))

        feed_dict = {images: batch}
        for name in a:
            try:
                prob_tensor = graph.get_tensor_by_name(name)
                prob = sess.run(prob_tensor, feed_dict=feed_dict)
                prob = np.array(prob)
                print(name, prob.shape)
            except BaseException:
                prob_tensor = graph.get_operation_by_name(name)
                prob = sess.run(prob_tensor.outputs, feed_dict=feed_dict)
                prob = np.array(prob)
                print(name, prob.shape)
                # print(prob)


                # try:
                #     print(name, prob.shape)
                # except BaseException:
                #     pass


if __name__ == '__main__':
    main()
