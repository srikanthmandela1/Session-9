# Session-9


import tensorflow as tf

# Define the input shape
input_shape = (32, 32, 3)

# Define the model architecture
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.BatchNormalization(),

    # Second convolutional layer
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.BatchNormalization(),

    # Third convolutional layer
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.BatchNormalization(),

    # Flatten the output from the convolutional layers
    tf.keras.layers.Flatten(),

    # First fully connected layer
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    # Second fully connected layer
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()



import tensorflow as tf

# Define the input shape
input_shape = (32, 32, 3)

# Define the model architecture
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.BatchNormalization(),

    # Second convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.BatchNormalization(),

    # Third convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.BatchNormalization(),

    # Fourth convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    # Fifth convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    # Add a convolutional layer with 2x2 kernel and VALID padding
    tf.keras.layers.Conv2D(filters=48, kernel_size=(2,2), padding='VALID', activation='relu'),

    # Sixth convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    # Flatten the output from the convolutional layers
    tf.keras.layers.Flatten(),

    # First fully connected layer
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    # Second fully connected layer
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()



import tensorflow as tf

# Define the input shape
input_shape = (32, 32, 3)

# Define the model architecture
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.BatchNormalization(),

    # Second convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.BatchNormalization(),

    # Third convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.BatchNormalization(),

    # Fourth convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    # Fifth convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    # Apply Global Average Pooling
    tf.keras.layers.GlobalAveragePooling2D(),

    # Reshape the output to (1, 1, 48)
    tf.keras.layers.Reshape((1, 1, 48)),

    # Sixth convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    # Flatten the output from the convolutional layers
    tf.keras.layers.Flatten(),

    # First fully connected layer
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    # Second fully connected layer
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()


import tensorflow as tf

class ULTIMUS(tf.keras.layers.Layer):
    def __init__(self, units):
        super(ULTIMUS, self).__init__()
        self.units = units
        
        # Define K, Q, and V fully connected layers
        self.K = tf.keras.layers.Dense(units=self.units//6)
        self.Q = tf.keras.layers.Dense(units=self.units//6)
        self.V = tf.keras.layers.Dense(units=self.units//6)
        
        # Define output fully connected layer
        self.out = tf.keras.layers.Dense(units=self.units)
        
    def call(self, x):
        # Apply K, Q, and V to input
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        
        # Reshape K, Q, and V to 3D tensors
        k = tf.reshape(k, shape=(-1, self.units//6, 8))
        q = tf.reshape(q, shape=(-1, self.units//6, 8))
        v = tf.reshape(v, shape=(-1, self.units//6, 8))
        
        # Compute attention map
        a = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(float(self.units//6))
        a = tf.nn.softmax(a, axis=-1)
        
        # Compute output
        z = tf.matmul(a, v)
        z = tf.reshape(z, shape=(-1, self.units))
        out = self.out(z)
        
        return out
# Define the model architecture
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    
    # Second convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    
    # Third convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    
    # Global average pooling
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # ULTIMUS block
    ULTIMUS(units=48),
    
    # Output layer
    tf.keras.layers.Dense(units=num_classes, activation='softmax')
])
import tensorflow as tf

# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define input shape
input_shape = x_train.shape[1:]

# Define ULTIMUS block
class ULTIMUS(tf.keras.layers.Layer):
    def __init__(self, units):
        super(ULTIMUS, self).__init__()
        self.units = units
        
        # Define K, Q, and V fully connected layers
        self.K = tf.keras.layers.Dense(units=self.units//6)
        self.Q = tf.keras.layers.Dense(units=self.units//6)
        self.V = tf.keras.layers.Dense(units=self.units//6)
        
        # Define output fully connected layer
        self.out = tf.keras.layers.Dense(units=self.units)
        
    def call(self, x):
        # Apply K, Q, and V to input
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        
        # Reshape K, Q, and V to 3D tensors
        k = tf.reshape(k, shape=(-1, self.units//6, 8))
        q = tf.reshape(q, shape=(-1, self.units//6, 8))
        v = tf.reshape(v, shape=(-1, self.units//6, 8))
        
        # Compute attention map
        a = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(float(self.units//6))
        a = tf.nn.softmax(a, axis=-1)
        
        # Compute output
        z = tf.matmul(a, v)
        z = tf.reshape(z, shape=(-1, self.units))
        out = self.out(z)
        
        return out

# Define the model architecture
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    
    # Second convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    
    # Third convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    
    # Global average pooling
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # ULTIMUS block
    ULTIMUS(units=48),
    
    # Output layer
    tf.keras.layers.Dense(units=num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# Train the model
batch_size = 128
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))


# Define the model architecture
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    
    # Second convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    
    # Third convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    
    # Global average pooling
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # Ultimus block 1
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], tf.transpose(x[1])) / tf.sqrt(float(8))),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    
    # Ultimus block 2
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], tf.transpose(x[1])) / tf.sqrt(float(8))),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    
    # Ultimus block 3
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], tf.transpose(x[1])) / tf.sqrt(float(8))),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    
    # Ultimus block 4
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], tf.transpose(x[1])) / tf.sqrt(float(8))),
    tf.keras.layers.Dense(units=48*8, activation='relu'),
    
    # Output layer
    tf.keras.layers.Dense(units=num_classes, activation='softmax')
])


import tensorflow as tf

# Define input shape
input_shape = (32, 32, 3)

# Define the model architecture
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    # Second convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    # Third convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    # Global Average Pooling
    tf.keras.layers.GlobalAveragePooling2D(),
    # Ultimus block (repeated 4 times)
    tf.keras.layers.Dense(units=8*48, activation='relu'),
    tf.keras.layers.Dense(units=8*48, activation='relu'),
    tf.keras.layers.Dense(units=8*48, activation='relu'),
    tf.keras.layers.Lambda(lambda x: tf.reshape(x, shape=(-1, 8, 48))),
    tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1])),
    tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1])/tf.math.sqrt(tf.cast(tf.shape(x[1])[-1], tf.float32))),
    tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1])),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=48, activation='relu'),
    # Final FC layer to output class scores
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()



import tensorflow as tf

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the input shape
input_shape = (32, 32, 3)

# Define the model architecture
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    # Second convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    # Third convolutional layer
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # Global Average Pooling
    tf.keras.layers.GlobalAveragePooling2D(),
    # Ultimus block
    tf.keras.layers.Dense(units=48, activation='relu'),
    tf.keras.layers.Dense(units=48, activation='relu'),
    tf.keras.layers.Dense(units=8),
    # Attention mechanism
    tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0,2,1])),
    tf.keras.layers.Dense(units=8, activation='softmax'),
    tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0,2,1])),
    # Final FC layer
    tf.keras.layers.Dense(units=48, activation='relu'),
    # Output layer
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Define the OCP optimizer
class OCP(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.9, epsilon=1e-7, name="OCP", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("momentum", momentum)
        self.epsilon = epsilon or tf.keras.backend.epsilon()

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        momentum_var = self.get_slot(var, "momentum")
        momentum_t = self._get_hyper("momentum", var_dtype)

        momentum_t = tf.cast(momentum_t, var_dtype)
        momentum_var_t = momentum_var * momentum_t - grad * lr_t
        var_t = var + momentum_var_t

        momentum_var.assign(momentum_var_t)

        return var.assign(var_t)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

# Compile the model
model.compile(optimizer=OCP(), loss=''



# Train the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=24,
                    validation_data=(test_images, test_labels))



import matplotlib.pyplot as plt

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()



