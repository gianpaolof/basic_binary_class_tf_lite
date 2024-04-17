My first example of a binary classification for Android with tflite.

The app has simply two edit button where one can input hour and day, and a text view to display the result.
The idea behind the dataset: from 8 am to 00 pm and from monday to sunday, suppose one collects the status of a switch
and wants to understand if there is a pattern. in my case the scattered plot of the data shows this:
![immagine](https://github.com/gianpaolof/basic_binary_class_tf_lite/assets/6586650/a2b7cfea-23cf-42ac-a864-3e8fd3c44375)

The green circle is the one where the status of the switch is on.

Running the neural network model, I can plot the decision boundary:
![immagine](https://github.com/gianpaolof/basic_binary_class_tf_lite/assets/6586650/9c897ea4-51b8-42fe-9d1e-456edd56f961)

I think the model did a decent job in finding a pattern (as far as I know given my ultra-basic knowledge in this field)

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
sample_size = 400

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


save and export 

# Import necessary libraries
import tensorflow as tf
# Save the trained model (optional)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save the TFLite model
with open("model.tflite", "wb") as f:
  f.write(tflite_model)


#load
# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


# Your sample data
hour = 22 
day = 7

# Create NumPy array with correct shape and data type
mysample = np.array([hour, day], dtype=np.int32).reshape(1, 2) 

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], mysample)

interpreter.invoke()
tflite_predictions = interpreter.get_tensor(output_details[0]['index'])

prediction = model.predict(mysample)

print(tflite_predictions)
print(prediction)


#decision boundary
# define bounds of the domain

# define bounds of the domain
min1, max1 = X_train.iloc[:, 0].min()-1, X_train.iloc[:, 0].max()+1
min2, max2 = X_train.iloc[:, 1].min()-1, X_train.iloc[:, 1].max()+1

print(min1, max1)
print(min2, max2)

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

# reshape the predictions back into a grid
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


