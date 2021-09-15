import tensorflow as tf

def build(input_shape, spec, name='a2c', activation=tf.tanh):

        num_layers = spec['layer']
        num_neuron = spec['neuron_per_layer']

        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        h = x_input
        for i in range(num_layers):
          # TODO: elu & xavier init
          h = tf.keras.layers.Dense(units=num_neuron, name='mlp_fc{}'.format(i), activation=activation)(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h], name=name)
        return network