from bokeh.io import curdoc
import joblib
from bokeh.layouts import gridplot, row, column, layout
from bokeh.models import Select, ColumnDataSource, Div, DateRangeSlider, HoverTool, Image, Slider, RadioButtonGroup
from bokeh.models.widgets import Button
from bokeh.events import ButtonClick
from bokeh.plotting import figure, show, output_file
from src.mnist import *
import numpy as np
import pandas as pd

n_cols = 28 # Static for MNIST

test_images = pd.read_csv('data/test.csv')

#test_images = test_images.iloc[:, 1:]
number_of_test_images, _ = test_images.shape


def generate_training_set():
    labels, images = load_data()
    images = apply_threshold(images, threshold_slider.value)

    return labels, images


def start_training(event):
    labels, images = generate_training_set()
    x_train, y_train, x_test, y_test = setup_train_test_split(images, labels, test_slider.value / 100)
    model = setup_network(dense_layer_amount_slider.value)
    model = optimize_compile(model, 'adam', 0.01)
    model = train_model(model, x_train, y_train, 10, 5)
    evaluation = evaluate_model(model, x_test, y_test)
    import joblib
    joblib.dump(model, 'model.pkl')


def predict(event):
    try:
        inference = joblib.load('model.pkl')
    except FileNotFoundError:
        return False

    test_img = test_images.iloc[image_selector.value].values
    image_for_plot = test_img
    image_for_plot = image_for_plot.reshape((28, 28))
    image_for_plot = np.flipud(image_for_plot)
    test_img[test_img < threshold_slider.value] = 0
    test_img[test_img >= threshold_slider.value] = 1

    #Xp = Xp / 255
    #Xp = Xp.reshape(-1, n_cols)
    test_img = test_img.reshape(1, 784)
    predictions = inference.predict_proba(test_img)
    print(predictions)
    prediction_image.image(image=[image_for_plot], x=0, y=0, dw=28, dh=28)
    prediction_image.title.text = "image nr. " + str(image_selector.value)

    certainty_score = predictions*100
    # Certainty
    certainty_source.data = dict(
        number=[i for i in range(10)],
        certainty=certainty_score
    )


prediction_image = figure()
prediction_image.x_range.range_padding = prediction_image.y_range.range_padding = 0
img = np.zeros(28*28).reshape((28, 28))
prediction_image.image(image=[img], x=0, y=0, dw=28, dh=28)

# Prediction hbar graph
certainty_source = ColumnDataSource(data=dict(number=[], certainty=[]))

certainty = figure(title='Certainty score', plot_width=800, plot_height=800, sizing_mode='scale_both')
certainty.hbar(y='number', right='certainty', height=0.4, source=certainty_source)


# Define controls for training
threshold_slider = Slider(title='Pixel Threshold', start=0, end=255, value=170, step=1)
optimizer_button = RadioButtonGroup(labels=['adam', 'sgd'], active=0)
dense_layer_amount_slider = Slider(title='Dense Layers', start=1, end=10, value=3, step=1)
epochs_slider = Slider(title='Epochs', start=10, end=200, value=30, step=10)
test_slider = Slider(title='Test percentage', start=0, end=100, value=30, step=5)

# Defing training button
train_button = Button(label='Start training', button_type='success')
train_button.on_event(ButtonClick, start_training)

# Controls for predictions
image_selector = Slider(title='Image name', start=0, end=number_of_test_images, value=0, step=1)
predict_button = Button(label='Predict', button_type='success')
predict_button.on_event(ButtonClick, predict)

training_controls = [threshold_slider, test_slider, dense_layer_amount_slider, epochs_slider, optimizer_button, train_button]
prediction_controls = [image_selector, predict_button]

# Create the layout
training_inputs = column(*training_controls, width=600, height=100)
training_inputs.sizing_mode = 'fixed'
prediction_inputs = column(*prediction_controls, width=600, height=100)
prediction_inputs.sizing_mode = 'fixed'
prediction_layout = column([prediction_inputs, prediction_image])

l = layout([
    [training_inputs, prediction_layout, certainty],
    ],
    sizing_mode='scale_both')

curdoc().add_root(l)
curdoc().title = 'MNIST'
