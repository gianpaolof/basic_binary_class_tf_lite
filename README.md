My first example of a binary classification for Android with tflite.

The app has simply two edit button where one can input hour and day, and a text view to display the result.
The idea behind the dataset: from 8 am to 00 pm and from monday to sunday, suppose one collects the status of a switch
and wants to understand if there is a pattern. in my case the scattered plot of the data shows this:
![immagine](https://github.com/gianpaolof/basic_binary_class_tf_lite/assets/6586650/a2b7cfea-23cf-42ac-a864-3e8fd3c44375)

The green circle is the one where the status of the switch is on.

Running the neural network model, I can plot the decision boundary:
![immagine](https://github.com/gianpaolof/basic_binary_class_tf_lite/assets/6586650/9c897ea4-51b8-42fe-9d1e-456edd56f961)

I think the model did a decent job in finding a pattern (as far as I know given my ultra-basic knowledge in this field)
in the decision boundary plot indeed, it is possible to see that the probability of having the switch set to on increases toward
the points where the dataset shows the green circle (in the boundary plot the circle is now orange). On the ight there is the legend
explaining the values of the contour plot

When running the app, the user can input hour/day and get class 0 or class 1 (meaning probability 0 or 1 to find the switch on/off in thet time)

the model:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
hmin=8
hmax=24
daymin=1
daymax=7
# Define ranges for age and income
hours_range = (hmin, hmax)  # Minimum and maximum age (inclusive)
day_range = (daymin, daymax)  # Minimum and maximum income (inclusive)

# Sample size
sample_size = 800

# Generate uniformly distributed random numbers
hr = np.random.randint(low=hours_range[0], high=hours_range[1] + 1, size=sample_size)
day = np.random.randint(low=day_range[0], high=day_range[1] + 1, size=sample_size)

# Create a DataFrame 
data = pd.DataFrame({'hr': hr, 'day': day})


def assign_status(hr, day, center_hr, center_day, radius):
    distance_from_center = ((hr - center_hr)**2 + (day - center_day)**2)**0.5 
    return 1 if distance_from_center <= radius else 0

# Apply the function (adjusting the parameters)
data['status'] = data.apply(lambda row: assign_status(row['hr'], row['day'], 20, 5, 3), axis=1) 


colors = ['red' if row['status'] == 0 else 'green' for index, row in data.iterrows()]
plt.scatter(data['hr'], data['day'], alpha=0.7, c=colors)

# Add labels and title
plt.xlabel('hours')
plt.ylabel('days')
plt.title('Scatter Plot of hrs vs days and status Color')

# Add grid lines (optional)
plt.grid(True)

# Split data into features (X) and labels (y)
X = data[['hr', 'day']]
y = data['status']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 

model = keras.Sequential([
    layers.Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)) ,

    layers.Dense(4, activation='relu'),  # Hidden layer
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])



model.compile(optimizer='adam',
              loss='binary_crossentropy', 
              metrics=['accuracy']) 
history = model.fit(X_train, y_train, epochs=100, batch_size=6, validation_split=0.2, verbose=0)


test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)




```

    7/7 [==============================] - 0s 2ms/step - loss: 0.2334 - accuracy: 0.9000
    Test Accuracy: 0.8999999761581421
    


    
![png](output_0_1.png)
    



```python
#let's try to plot the decision boundary

# define bounds of the domain
min1, max1 = X_train.iloc[:, 0].min()-1, X_train.iloc[:, 0].max()+1
min2, max2 = X_train.iloc[:, 1].min()-1, X_train.iloc[:, 1].max()+1


# define the x and y scale
x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)

# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)

# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))


# make predictions for the grid
yhat = model.predict(grid)


zz = yhat.reshape(xx.shape)



c = plt.contourf(xx, yy, zz, cmap='RdBu')
# add a legend, called a color bar
plt.colorbar(c)

cmap = plt.cm.get_cmap('tab10') 
for class_value in range(2):
    row_ix = np.where(y == class_value)[0]  # Get array of indices
    for index in row_ix:  
        age = X.iloc[index, 0]  # Access 'age' using index
        income = X.iloc[index, 1]  # Access 'income' using index
        color = cmap(class_value) 
        plt.scatter(age, income, c=color) 


```

    450/450 [==============================] - 1s 1ms/step
    

    C:\Users\gianp\AppData\Local\Temp\ipykernel_6644\3112694210.py:35: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed two minor releases later. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap(obj)`` instead.
      cmap = plt.cm.get_cmap('tab10')
    C:\Users\gianp\AppData\Local\Temp\ipykernel_6644\3112694210.py:42: UserWarning: *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2D array with a single row if you intend to specify the same RGB or RGBA value for all points.
      plt.scatter(age, income, c=color)
    


    
![png](output_1_2.png)
    



```python
#custom model

class OnDeviceTrainableModel(keras.Model):

    def __init__(self):
        super(OnDeviceTrainableModel, self).__init__()

        # Updated model structure for your data
        self.model = tf.keras.Sequential([
         tf.keras.layers.Dense(8, activation='relu', input_shape=(2,), kernel_regularizer=keras.regularizers.l2(0.01)),  # Input shape for 'hr' and 'day'
         tf.keras.layers.Dense(4, activation='relu'), 
         tf.keras.layers.Dense(1, activation='sigmoid')   # Output for binary 'status'
        ])

        # Compilation - assuming a binary classification task
        self.model.compile(
            optimizer='adam',  # Updated optimizer
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics = ['accuracy']
        )

    @tf.function(input_signature=[
        tf.TensorSpec([None, 2], tf.float32),  # For your 'hr' and 'day'
    ])  
    def infer(self, inputs):  # Changed the input name for clarity
        logits = self.model(inputs)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {
            "output": probabilities,
            "logits": logits
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
          restored = tf.raw_ops.Restore(
              file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
              name='restore')
          var.assign(restored)
          restored_tensors[var.name] = restored
        return restored_tensors

    @tf.function(input_signature=[  # Adjust shapes/dtypes if needed
            tf.TensorSpec([None, 2], tf.float32),  # For your 'hr' and 'day'
            tf.TensorSpec([None, ], tf.float32),  # For your 'status'
        ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
          prediction = self.model(x)
          loss = self.model.loss(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss": loss}
        return result
          
    def call(self, inputs):
        x = self.model(inputs)  # Apply the entire Sequential model 
        return x
        
model1 = OnDeviceTrainableModel()
xt = tf.convert_to_tensor(X_train, dtype=tf.float32)  # Cast hours to float32
yt = tf.convert_to_tensor(y_train, dtype=tf.float32) 


BATCH_SIZE = 6  # Adjust batch size as needed
val_split = 0.2
val_size = int(val_split * len(xt)) 

train_ds = tf.data.Dataset.from_tensor_slices((xt[:-val_size], yt[:-val_size])).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((xt[-val_size:], yt[-val_size:])).batch(BATCH_SIZE)

NUM_EPOCHS=100
losses = np.zeros([NUM_EPOCHS])
for i in range(NUM_EPOCHS):
    for x_batch, y_batch in train_ds:  # Iterate over batches in the dataset
        result = model1.train(x_batch, y_batch)  
        
    losses[i] = result['loss']

        #if (i + 1) % 20 == 0:
        #    print(f"Finished {i+1} epochs")
        #    print(f"  loss: {losses[i]:.3f}")
# Save the trained weights to a checkpoint.
#m.save('/tmp/model.ckpt')



```


```python
SAVED_MODEL_DIR = "saved_model"
tf.saved_model.save(
    model1,
    SAVED_MODEL_DIR,
    signatures={
        'train':
            model1.train.get_concrete_function(),
        'infer':
            model1.infer.get_concrete_function(),
        'save':
            model1.save.get_concrete_function(),
        'restore':
            model1.restore.get_concrete_function(),
    })

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()
```

    WARNING:tensorflow:Skipping full serialization of Keras layer <__main__.OnDeviceTrainableModel object at 0x0000020AF5874E20>, because it is not built.
    INFO:tensorflow:Assets written to: saved_model\assets
    

    WARNING:absl:Importing a function (__inference_internal_grad_fn_72018) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.
    WARNING:absl:Importing a function (__inference_internal_grad_fn_72058) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.
    


```python
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()


infer = interpreter.get_signature_runner("infer")

```


```python
#let's try to plot the decision boundary

# define bounds of the domain
min1, max1 = X_train.iloc[:, 0].min()-1, X_train.iloc[:, 0].max()+1
min2, max2 = X_train.iloc[:, 1].min()-1, X_train.iloc[:, 1].max()+1


# define the x and y scale
x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)

# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)

# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

t1 = tf.convert_to_tensor(r1)
t2 = tf.convert_to_tensor(r2)
grid2 = tf.concat([t1, t2], axis=1)
grid2 = tf.cast(grid2, tf.float32) 

yhat2 = infer(inputs=grid2)['logits']


#uncomment to obtain the plots of the tflite model (infer)
zz = yhat2.reshape(xx.shape)



c = plt.contourf(xx, yy, zz, cmap='RdBu')
# add a legend, called a color bar
plt.colorbar(c)

cmap = plt.cm.get_cmap('tab10') 
for class_value in range(2):
    row_ix = np.where(y == class_value)[0]  # Get array of indices
    for index in row_ix:  
        age = X.iloc[index, 0]  # Access 'age' using index
        income = X.iloc[index, 1]  # Access 'income' using index
        color = cmap(class_value) 
        plt.scatter(age, income, c=color) 


```

    C:\Users\gianp\AppData\Local\Temp\ipykernel_6644\3976961887.py:36: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed two minor releases later. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap(obj)`` instead.
      cmap = plt.cm.get_cmap('tab10')
    C:\Users\gianp\AppData\Local\Temp\ipykernel_6644\3976961887.py:43: UserWarning: *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2D array with a single row if you intend to specify the same RGB or RGBA value for all points.
      plt.scatter(age, income, c=color)
    


    
![png](output_5_1.png)
    



```python

```


