import tensorflow as tf
import numpy as np

HOLES = 5
DAYS = 5

LR = 0.01

EPOCHS = 1000


def getConns(holes):
    conns = []
    low = -1
    high = 1
    for i in range(holes):
        conn = [0]*holes
        low = i - 1
        high = i + 1
        if low < 0:
            conn[high] = 1.0
        elif high >= holes:
            conn[low] = 1.0
        else:
            conn[high] = 0.5
            conn[low] = 0.5
        conns.append(conn)
    return conns


if __name__ == '__main__':
    init = tf.random_uniform_initializer(minval=0.99, maxval=1.01)
    all_weights = [tf.Variable(init((HOLES,), dtype=tf.dtypes.float64))
                   for _ in range(DAYS)]
    '''
    all_weights = [tf.Variable(tf.ones((HOLES,), dtype=tf.dtypes.float64)) for _ in range(DAYS)]
    '''

    connections_list = [tf.constant(conn, dtype=tf.dtypes.float64)
                        for conn in getConns(HOLES)]
    connections = tf.convert_to_tensor(connections_list)

    optim = tf.keras.optimizers.Adam(learning_rate=LR)

    pot_init = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
    for epoch in range(EPOCHS):
        '''
        weight_tensors = [tf.convert_to_tensor(weights)
                          for weights in all_weights]
        '''
        with tf.GradientTape(persistent=True) as gt:
            for weights in all_weights:
                gt.watch(weights)

            potential = pot_init((HOLES,), dtype=tf.dtypes.float64)
            potential = tf.nn.softmax(potential)
            '''
            potential = tf.ones((HOLES,), dtype=tf.dtypes.float64)
            '''
            potential *= tf.constant(100, dtype=tf.dtypes.float64)
            for weights in all_weights:
                # Remove some potential
                sf = tf.nn.softmax(weights)
                coefs = tf.ones((HOLES,), dtype=tf.dtypes.float64) - sf 
                potential = potential * coefs
                '''
                idxHigh = tf.math.argmax(weights)
                mask_list = [0]*HOLES
                mask_list[idxHigh] = 1
                mask = tf.cast(mask_list, tf.dtypes.float64)
                potential *= mask
                '''

                # Distribute potential to next day
                potential = potential * connections

                potential = tf.reduce_sum(potential, axis=1)
                # potential = tf.linalg.matrix_transpose(potential)
                # print(potential)
                '''
                for conn in connections:
                    potential = potential * conn
                '''
            '''
            closeness = tf.convert_to_tensor(1, dtype=tf.dtypes.float64)
            for weights in all_weights:
                sf = tf.nn.softmax(weights)
                closeness = tf.minimum(tf.reduce_max(sf), closeness)
            close_penalty = tf.constant(1, dtype=tf.dtypes.float64) - closeness
            close_penalty *= tf.constant(0.5, dtype=tf.dtypes.float64)
            close_penalty += tf.constant(0.5, dtype=tf.dtypes.float64)

            loss = tf.reduce_sum(potential) * close_penalty
            '''
            loss = tf.reduce_sum(potential)

        gradients = []
        for weights in all_weights:
            grad = gt.gradient(loss, weights)
            gradients.append(grad)

        optim.apply_gradients(zip(gradients, all_weights))

        if epoch % 100 == 0:
            print(f'epoch: {epoch}    loss: {loss.numpy()}')

    for weights in all_weights:
        w_arr = tf.nn.softmax(weights).numpy() * 100
        print(' '.join([f'{int(w>50)}' for w in w_arr]))



            
