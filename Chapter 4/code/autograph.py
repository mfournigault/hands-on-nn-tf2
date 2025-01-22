import tensorflow as tf

# First solution to solve the state of variables when a graph is created (decorator tf.function), passing 
# the variable as an argument to the function
@tf.function
def f(b):
    a = tf.constant([[10, 10], [11., 1.]])
    x = tf.constant([[1., 0.], [0., 1.]])
    # b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    return y
    # for i in range(10):
    #     print(i)

print("--- First solution: passing the variable as an argument to the function ---")
var = tf.Variable(12.)
print(f(var))
print(f(13))
# print(f(tf.constant(1))) # Error, can't process int32 with float32 for the y computation

# Second solution to solve the state of variables when a graph is created (decorator tf.function),
# breaking the function scope, declare the variable as a private attribute of the class, and make the class callable

class F:
    def __init__(self):
        self._b = None

    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x = tf.constant([[1., 0.], [0., 1.]])
        if self._b is None:
            self._b = tf.Variable(12.)
        y = tf.matmul(a, x) + self._b
        return y

f = F()

print("--- Second solution: breaking the function scope ---")
print(f())

