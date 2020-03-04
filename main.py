import tensorflow as tf
import numpy as np

HOLES = 4
DAYS = 4
LR = 1.0
EPOCHS = 3000
POTENTIAL_PER_HOLE = 100.0


def getConns(holes):
    conns = []
    low = -1
    high = 1
    for i in range(holes):
        conn = [0]*holes
        low = i - 1
        high = i + 1
        if low >= 0:
            conn[low] = 1.0
        if high < holes:
            conn[high] = 1.0
        total = sum(conn)
        conn = [x/total for x in conn]
        conns.append(conn)
    return conns


@tf.function
def propagate(all_weights_stack, potential):
    all_weights = tf.unstack(all_weights_stack)
    potential_log = []
    for weights in all_weights:
        # Remove some potential
        sf = tf.math.sigmoid(weights)
        coefs = tf.ones((HOLES,)) - sf
        potential = potential * coefs

        # Distribute potential to next day
        potential = potential * connections
        potential = tf.reduce_sum(potential, axis=1)
        potential_log.append(potential)
    return potential, tf.stack(potential_log)


@tf.function
def calc_loss(all_weights_stack, potential_log):
    # Calculate check penalty
    bounded = tf.math.sigmoid(all_weights_stack)
    day_sums = tf.reduce_sum(bounded, axis=1)
    check_penalties = tf.square(tf.constant(1.0) - day_sums*tf.constant(3.0))
    # check_penalties = tf.square(day_sums)
    check_penalty = tf.reduce_sum(check_penalties)

    # Calculate potentials penalty
    pot_loss = tf.reduce_sum(potential_log[-1])
    total_loss = pot_loss + check_penalty
    return total_loss, pot_loss


def show_policy(all_weights_stack):
    all_weights = tf.unstack(all_weights_stack)
    toggle = 0
    for weights in all_weights:
        toggle += 1
        w_arr = tf.math.sigmoid(weights).numpy()
        # Don't worry about it, it's fine
        lines = [f"{0 if w>0.5 else '.,'[(idx+toggle)%2]}"
                 for idx, w in enumerate(w_arr)]
        print(' '.join(lines))


if __name__ == '__main__':
    init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
    all_weights = tf.Variable([init((HOLES,)) for _ in range(DAYS)])

    connections_list = [tf.constant(conn) for conn in getConns(HOLES)]
    connections = tf.convert_to_tensor(connections_list)

    optim = tf.keras.optimizers.Adam(learning_rate=LR)
    pot_init = tf.random_uniform_initializer(minval=0.0, maxval=0.8)

    counter = 1
    counter_max = 300
    no_check_penalty_share = 0.0
    for epoch in range(EPOCHS):
        with tf.GradientTape(persistent=True) as gt:
            potential = pot_init((HOLES,))
            potential = tf.round(potential)
            potential *= tf.constant(POTENTIAL_PER_HOLE)
            potential, pot_log = propagate(all_weights, potential)
            loss, pot_loss = calc_loss(all_weights, pot_log)

        # Calculate gradients and update weights
        if counter > no_check_penalty_share*counter_max:
            gradients = gt.gradient(loss, all_weights)
        else:
            gradients = gt.gradient(pot_loss, all_weights)

        optim.apply_gradients([(gradients, all_weights)])

        # Calculate validation loss
        potential = tf.ones((HOLES,))
        potential *= tf.constant(POTENTIAL_PER_HOLE)
        potential, pot_log = propagate(all_weights, potential)
        val_loss, pot_loss = calc_loss(all_weights, pot_log)

        counter += 1
        if counter >= counter_max:
            counter = 1

        if epoch % 100 == 0:
            msg = f'epoch: {epoch:<8}\t'
            msg += f'val_loss: {val_loss.numpy():<6.5}\t'
            msg += f'pot_loss: {pot_loss.numpy():<6.5}'
            print(msg)

    print("RESULT:")
    show_policy(all_weights)
