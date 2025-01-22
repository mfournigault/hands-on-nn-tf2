import tensorflow as tf


@tf.function
def f():
    x = 0
    for i in range(10):
        print(i)
        x += i
    return x

print("First call")
f()
print("Second call")
f()
print("--- Autograph code ---")
print(tf.autograph.to_code(f.python_function))
