# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()   
import numpy as np
from utils import gen_ex, interpret

def wordtree_loss(logits, labels, word_tree, epsilon = 1e-5):
    '''
    Builds the wordtree style loss function as described in YOLO9000
    (https://arxiv.org/abs/1612.08242)

    Args:
        logits (tf.Tensor): Classification logits.
        labels (tf.Tensor): The one hot tensor of the ground truth labels.
        word_tree (dict): Dictionary of dictionaries showing the relationship between the classes.
        epsilon (float, optional): Epsilon term added to make the softmax cross entropy stable. Defaults to 1e-5.

    Returns:
        loss: Tensor of shape (batch_size, ), giving the loss for each example.
        raw_probs: The probability for each class (given its parents).
    '''
    _, n = _get_dict_item(word_tree)
    n_flat = [len(n)] + list(_flatten(n))
    parents, _, _ = _get_idxs(n_flat)
    
    subsoftmax_idx = np.cumsum([0] + n_flat, dtype = np.int32)
    
    raw_probs = tf.concat([
        tf.nn.softmax(logits[:, subsoftmax_idx[i]: subsoftmax_idx[i + 1]]) \
        for i in range(len(n_flat))], 1)
    probs = tf.concat([tf.reduce_prod(tf.gather(raw_probs, p, axis = 1),
                                  axis = 1,
                                  keepdims = True) for p in parents], 1)
    
    loss = tf.reduce_sum(-tf.log(probs + epsilon) * labels, 1)
    return loss, raw_probs

def _get_dict_item(d):
    l = []; n = []
    if type(d) is dict:
        keys, items = zip(*sorted(d.items(), key = lambda x:x[0]))
        l += list(keys)
        for key, item in zip(keys, items):
            _l, _n = _get_dict_item(item)
            l += [key + '/' +i for i in _l]
            n += [_n]
    else:
        return d, 0
    return l, n

def _get_idxs(flat_tree_form):
    '''
    return parents, children, childless
      parents: list of lists of idxs of parents of each child
      children: list of lists of idxs of children of each parent
      childless: list of idxs of the childless
    '''
    parents = []
    children = [[] for _ in range(len(flat_tree_form) - 1)]
    childless = []
    mp = []
    p = []
    c = 0
    for n in flat_tree_form:
        for i in range(c, c + n): parents += [p + [i]]
        if len(p) > 0: children[p[-1]] = list(range(c, c+n))
        if n == 0:
            childless += [p[-1]]
            p[-1] += 1
            while p[-1] == mp[-1] and len(p) > 1:
                p.pop(-1); mp.pop(-1)
                p[-1] += 1
        else:
            p += [c]
            mp += [c + n]
        c += n
    return parents, children, childless

def _flatten(l):
    for i in l:
        if type(i) is list:
            yield len(i)
            for f in _flatten(i):
                yield f
        else:
            yield i


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    word_tree = {'animal': {'cat': {'big-cat': {'lion': '', 'tiger': ''},
                                    'small-cat': ''},
                            'dog': {'collie': '', 'dalmatian': '', 'terrier': ''},
                            'mouse': ''},
                 'elements': {'acid': {'h2so4': '', 'hcl': ''},
                              'base': {'strong': {'koh': '', 'naoh': ''},
                                       'weak': {'ch3nh2': '', 'nh3': '', 'nh4oh': ''}}}}
    
    class_list, n = _get_dict_item(word_tree)
    print("class_list", class_list)
    """
    class_list ['animal', 'elements', 'animal/cat', 'animal/dog', 'animal/mouse', 'animal/cat/big-cat', 'animal/cat/small-cat', 'animal/cat/big-cat/lion', 'animal/cat/big-cat/tiger', 'animal/dog/collie', 'animal/dog/dalmatian', 'animal/dog/terrier', 'elements/acid', 'elements/base', 'elements/acid/h2so4', 'elements/acid/hcl', 'elements/base/strong', 'elements/base/weak', 'elements/base/strong/koh', 'elements/base/strong/naoh', 'elements/base/weak/ch3nh2', 'elements/base/weak/nh3', 'elements/base/weak/nh4oh']
    """
    print("n", n)
    #n_flat [[[[0, 0], 0], [0, 0, 0], 0], [[0, 0], [[0, 0], [0, 0, 0]]]]
    
    num_root = len(n)
    print("num_root", num_root)
    # 2
    
    n_flat = [num_root] + list(_flatten(n))
    print("n_flat", n_flat)
    # n_flat [2, 3, 2, 2, 0, 0, 0, 3, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0]

    n_classes = len(n_flat) - 1
    print("n_classes", n_classes)
    # n_classes 23
    n_filters = 16
    
    parents, children, childless = _get_idxs(n_flat)
    print("parents", parents)
    # parents [[0], [1], [0, 2], [0, 3], [0, 4], [0, 2, 5], [0, 2, 6], [0, 2, 5, 7], [0, 2, 5, 8], [0, 3, 9], [0, 3, 10], [0, 3, 11], [1, 12], [1, 13], [1, 12, 14], [1, 12, 15], [1, 13, 16], [1, 13, 17], [1, 13, 16, 18], [1, 13, 16, 19], [1, 13, 17, 20], [1, 13, 17, 21], [1, 13, 17, 22]]
    print("children", children)
    # children [[2, 3, 4], [12, 13], [5, 6], [9, 10, 11], [], [7, 8], [], [], [], [], [], [], [14, 15], [16, 17], [], [], [18, 19], [20, 21, 22], [], [], [], [], []]
    print("childless", childless)
    # childless [7, 8, 6, 9, 10, 11, 4, 14, 15, 18, 19, 20, 21, 22]

    y_b = np.array(range(n_classes))
    print("y_b", y_b)
    # y_b [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]
    num_ex = len(y_b)
    print("num_ex", num_ex)
    # num_ex 23
    x_b = gen_ex(y_b, parents, children)
    print("x_b", x_b)
    """
x_b [[ 2.44496488e-01  3.72602288e-01  1.53213385e-01  2.40230428e-01
   3.07404543e-01  2.04694420e-01  6.02922900e-01  4.80634933e-01
  -1.11482451e-02  7.93737068e-01  4.08896210e-01  2.92296350e-01
   8.59654267e-01  7.91866711e-01  5.66827314e-01  3.44718595e-01]
 [ 5.77298703e-01  4.06972445e-01  4.61480960e-01  3.95394534e-01
   1.66511961e-01  2.90673080e-01  2.84326914e-01  5.72647414e-01
   6.62204716e-01  8.60303467e-02  4.27175530e-01  7.82965447e-01
   4.50060127e-01  8.12491604e-01  6.18255071e-01  8.73056857e-01]
 [ 2.56731083e-01  4.03983217e-01 -2.90214660e-02  1.44002424e-01
   6.11495732e-01  3.92966077e-01  6.28804774e-01  3.75299608e-01
   1.57206445e-01  6.61860659e-01  5.42108074e-01  3.25243537e-01
   7.75614392e-01  3.98256290e-01  2.99290398e-01  3.64907613e-01]
 [ 1.30642175e-01  3.30770911e-01  4.00944508e-01  4.56777513e-01
   4.48007719e-01  8.48206649e-02  3.12433410e-01  6.79510970e-01
   3.53464114e-01  7.45527630e-01  3.86429153e-01 -1.39107027e-01
   7.22400636e-01  3.85900276e-01  1.57612104e-01  7.69214083e-01]
 [ 8.52925958e-02  1.83571826e-03  4.28303567e-02  9.96277835e-02
  -4.91149656e-02  7.12678166e-02  1.11334035e-01 -2.15367459e-01
  -4.16111482e-02 -1.07089699e-01  2.21138805e-02 -1.12305712e-01
  -1.05075796e-01  1.01207905e-01  1.54371643e-01 -4.02114889e-02]
 [ 2.11474910e-01  2.68979744e-01  6.06116027e-01  2.53779461e-01
   5.04486517e-01  4.82512877e-01 -2.25776001e-02  6.35739778e-01
   1.43425765e-01  1.05571455e-01  8.00610703e-01  5.22297991e-01
   6.98880888e-02  4.41143478e-02  9.52144274e-01  4.82997407e-01]
 [ 5.62173162e-03  1.77699478e-01 -8.25590883e-02  3.51128532e-01
   6.90044008e-01  1.84500832e-01  6.24420931e-01  3.11894772e-01
   6.65827901e-02  6.55564534e-01  3.67770488e-01  1.45341089e-01
   8.75973465e-01  4.81865136e-01  3.58320519e-01  2.38077590e-01]
 [ 4.91222720e-02  1.27477896e-02  4.28087110e-01  3.79277413e-02
   4.41840565e-01  3.63357988e-02  4.68503240e-01  3.31068012e-01
   7.24735381e-01  5.02692229e-01  2.00983930e-01  1.65136540e-02
   7.68029571e-01  4.18251847e-01  2.79921729e-01 -1.18742027e-01]
 [ 2.19981167e-02  2.39921480e-01  5.10819999e-01  9.73519787e-02
   7.06484661e-01  4.57594331e-01  2.50139612e-02  6.42250503e-01
   2.68785426e-01  1.39568494e-03  7.06943503e-01  4.65285644e-01
   8.11758118e-02  5.21248834e-02  7.22606354e-01  5.56846702e-01]
 [-2.62230646e-02  3.44689103e-01  2.95784168e-01  3.21676397e-01
   5.31564128e-01  1.51578572e-01  5.40184215e-01  7.21376215e-02
   4.43552780e-01  2.93382745e-01  7.22183213e-01  4.85767401e-01
  -7.12773804e-04  8.39315102e-01  4.80901517e-01  1.15342387e-01]
 [ 2.12284298e-01  1.66136893e-01  1.13732887e-01  4.63312703e-01
   4.46597691e-01 -9.35014567e-02  3.84041728e-01  7.56257502e-01
   2.07831284e-01  4.62222127e-01  3.50518898e-01  1.53926172e-01
   6.14140216e-01  4.23198245e-01  4.63633467e-02  8.08891422e-01]
 [ 2.97036519e-01  3.55967002e-01  3.91721345e-01  5.06379269e-01
   4.48162474e-01 -1.30497134e-02  3.35965079e-01  6.50893216e-01
   5.64080818e-02  6.67939943e-01  2.63028536e-01  6.21374099e-01
   4.97342791e-01  2.84236304e-02  6.24510379e-01  3.30524220e-01]
 [ 4.12713620e-01  7.79560167e-02  5.31191349e-01  4.59895799e-01
   5.59462603e-01  6.32530906e-01  7.73242156e-01  2.35715788e-02
   3.83994380e-01  6.04375507e-01  2.13555834e-01  4.74175418e-01
   3.13294233e-01  7.13691284e-01  1.41754682e-01  9.25381748e-01]
 [ 4.15791568e-01  2.77637381e-04  5.29097119e-01  2.05926685e-01
   3.32211007e-01  1.33464252e-01  3.34539868e-01  3.45174600e-01
   6.05207848e-01 -5.38979328e-02  2.51757725e-01  5.22729641e-01
   3.91258031e-02  6.17610239e-01  1.50369946e-01  7.87619506e-01]
 [ 2.44491242e-01  7.82069887e-02  2.93901527e-01  5.33831078e-01
   4.41702119e-01  7.19026126e-01 -6.97659438e-02  8.22954348e-02
   6.17789999e-01 -2.28127987e-02  3.94943143e-01  9.03001676e-01
   2.60404688e-01  8.24685412e-02  5.83899176e-01  3.00306753e-01]
 [ 4.75554921e-01  1.99135249e-01  6.97334671e-01  4.33108165e-01
   4.34028308e-01  6.53304121e-01  6.49902070e-01  1.00143858e-01
   3.11091385e-01  9.24776759e-01 -9.21794828e-02  5.20933610e-01
  -9.00189244e-02  7.87400545e-01  2.53636299e-01  9.88183157e-01]
 [ 4.72473304e-01  2.99025033e-01  5.27073378e-01  3.04163229e-01
   3.49202934e-01  2.62273004e-01  2.72828481e-01  5.56170638e-01
   6.94258787e-01  3.52129274e-01  5.71219330e-01  8.27249924e-01
   1.05995561e-01  7.99770892e-01  3.92223047e-01  8.38710460e-01]
 [ 4.66373961e-01  2.49304422e-01  3.86374849e-01  2.20509315e-01
   1.92512716e-01  3.13794743e-01  4.98197092e-02  1.94200816e-01
   3.19576779e-01  6.53873274e-01  1.34006542e-01  4.61266958e-01
   8.21919519e-01  2.84882143e-01  1.12373457e+00  3.53051506e-01]
 [ 4.01566205e-01  8.97003759e-02  3.47864698e-01  5.41749862e-02
   1.53377799e-01  2.05044684e-01  2.36578290e-01  2.66178618e-01
   5.39693772e-01  1.90099585e-02  3.37364980e-01  6.03693609e-01
   3.82443135e-01  6.39892002e-01  1.59583814e-01  9.20745595e-01]
 [ 4.53771778e-01  6.72707655e-02  5.70978602e-01  1.34639168e-01
   3.14697362e-01  4.22177243e-01  4.65049691e-01  6.61089968e-01
   7.63806456e-01  1.20968364e-01  6.42377939e-01  1.13248665e+00
   4.30263455e-01  1.04211102e+00  4.09837656e-01  9.23518092e-01]
 [ 6.52445153e-02  4.55876700e-01  1.30254380e-01  5.24280926e-01
   3.92034759e-01  3.23628705e-01  5.15480072e-01  6.35560554e-01
   1.34910334e-01  3.71693969e-01  4.90971860e-01  2.67285966e-01
   6.24379727e-01 -3.31775250e-03  5.78827109e-01  7.70454475e-02]
 [ 4.32488621e-01  3.46566093e-01 -3.01571040e-02  7.23026196e-02
   6.70397489e-01  6.90155856e-01  7.29071507e-01  7.58594160e-02
   1.23726528e-01  3.15400469e-01  7.45912831e-01  4.46900770e-01
   8.32441283e-01  1.23902904e-01  6.36861600e-01  2.07327191e-01]
 [ 5.36298910e-01  1.01519395e-01  2.77899732e-01  8.13834855e-02
   4.35646147e-02  1.37711959e-01 -1.14751613e-02  1.30808925e-01
   5.07752958e-01  7.00631327e-01 -8.52700044e-02  4.41157141e-01
   8.06249781e-01  4.52258104e-01  9.09073769e-01  6.02166591e-01]]
    """
    
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape = [None, 16])
    print("x", x)
    # x Tensor("Placeholder:0", shape=(?, 16), dtype=float32)

    y = tf.placeholder(tf.int32, shape = [None])
    print("y", y)
    # y Tensor("Placeholder_1:0", shape=(?,), dtype=int32)

    y_onehot = tf.one_hot(y, n_classes)
    print("y_onehot", y_onehot)
    """
    y_onehot Tensor("one_hot:0", shape=(?, 23), dtype=float32)
    hierarchical_loss.py:242: UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.
    d1 = tf.layers.dense(x, n_filters, activation = tf.nn.relu)
    """

    d1 = tf.layers.dense(x, n_filters, activation = tf.nn.relu)
    print("d1", d1)
    """
    d1 Tensor("dense/Relu:0", shape=(?, 16), dtype=float32)
    hierarchical_loss.py:244: UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.
    d2 = tf.layers.dense(d1, n_filters, activation = tf.nn.relu)
    """    
    d2 = tf.layers.dense(d1, n_filters, activation = tf.nn.relu)
    print("d2", d2)
    """
    d2 Tensor("dense_1/Relu:0", shape=(?, 16), dtype=float32)
    hierarchical_loss.py:247: UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.
    logits = tf.layers.dense(d2, n_classes)
    """
    
    logits = tf.layers.dense(d2, n_classes)
    print("logits", logits)
    """
    d2 = tf.layers.dense(d1, n_filters, activation = tf.nn.relu)
    d2 Tensor("dense_1/Relu:0", shape=(?, 16), dtype=float32)
    hierarchical_loss.py:247: UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.
    """
    
    _loss, raw_probs = wordtree_loss(logits = logits, labels = y_onehot, word_tree = word_tree)
    print("_loss", _loss)
    """
    _loss Tensor("Sum:0", shape=(?,), dtype=float32)
    """
    print("raw_probs", raw_probs)
    # raw_probs Tensor("concat:0", shape=(?, 23), dtype=float32)

    loss = tf.reduce_mean(_loss)
    print("loss", loss)
    # loss Tensor("Mean:0", shape=(), dtype=float32)

    train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
    print("train_op", train_op)
    # train_op name: "Adam"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(10):
            idxs = np.random.randint(0, num_ex, 8)
            _, l, rp = sess.run([train_op, loss, raw_probs], feed_dict = {x: x_b[idxs], y: y_b[idxs]})
            if (step + 1)%100 == 0:
                print('Step {}: loss = {:.03e}'.format(step + 1, l))
        """
        Step 100: loss = 7.009e-01
        Step 200: loss = 1.635e-02
        Step 300: loss = 9.109e-03
        Step 400: loss = 2.953e-03
        Step 500: loss = 2.156e-03
        Step 600: loss = 1.383e-03
        Step 700: loss = 2.651e-03
        Step 800: loss = 7.802e-04
        Step 900: loss = 1.043e-03
        Step 1000: loss = 9.031e-04

        """
    
        print('===TEST===')
        rp, = sess.run([raw_probs], feed_dict = {x: gen_ex(childless, parents, children, 0)})
        for r, gt in zip(rp, childless):
            pred_class = interpret(r, num_root, children)
            if not pred_class == gt:
                print('Truth: {}\t|\tPred: {}'.format(class_list[gt], class_list[pred_class]))
        """
        Если 1000 выше for step in range(1000):...
        ===TEST===
        Truth: elements/base/strong/naoh	|	Pred: elements/base/strong/koh
        """

        """
        Если 100 выше for step in range(100):...
        ===TEST===
        Truth: animal/cat/small-cat	|	Pred: animal/cat/big-cat/lion
        Truth: elements/acid/h2so4	|	Pred: elements/base/weak/nh3
        Truth: elements/base/weak/ch3nh2	|	Pred: animal/cat/big-cat/lion
        Truth: elements/base/weak/nh3	|	Pred: animal/cat/big-cat/lion
        """ 
        """
        Если 10 выше for step in range(10):...
        ===TEST===
        Truth: animal/cat/big-cat/lion	|	Pred: animal
        Truth: animal/cat/big-cat/tiger	|	Pred: elements/base/weak/ch3nh2
        Truth: animal/cat/small-cat	|	Pred: animal
        Truth: animal/dog/collie	|	Pred: animal
        Truth: animal/dog/dalmatian	|	Pred: elements/base/weak/ch3nh2
        Truth: animal/dog/terrier	|	Pred: elements/base/weak/ch3nh2
        Truth: animal/mouse	|	Pred: elements/base/strong/koh
        Truth: elements/acid/h2so4	|	Pred: elements/base/strong/koh
        Truth: elements/acid/hcl	|	Pred: elements/base/weak/ch3nh2
        Truth: elements/base/strong/koh	|	Pred: elements/base/weak/ch3nh2
        Truth: elements/base/strong/naoh	|	Pred: elements/base/weak/ch3nh2
        Truth: elements/base/weak/ch3nh2	|	Pred: animal
        Truth: elements/base/weak/nh3	|	Pred: elements/base/weak/ch3nh2
        Truth: elements/base/weak/nh4oh	|	Pred: elements/base/weak/ch3nh2
        """ 