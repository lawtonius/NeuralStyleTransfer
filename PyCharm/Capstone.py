from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import time
import os
app = Flask(__name__)

content_filename = ''
style_filename = ''

def add_time(filename):
    suffix = filename.rsplit('.')[1]
    filename = str(time.time()).replace('.', '_') + '.' + suffix
    return filename

def is_file_allowed(filename):
    if not "." in filename:
        return False
    suffix = filename.rsplit('.', 1)[1]
    if suffix.lower() in ['jpg', 'jpeg', 'npg']:
        return True
    else:
        return False

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype = np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        content_image = request.files['content']
        style_image = request.files['style']
        global content_filename, style_filename
        content_filename = 'c' + add_time(content_image.filename)
        style_filename = 's' + add_time(style_image.filename)
        if is_file_allowed(content_filename):
            content_image.save('./static/' + content_filename)
        if is_file_allowed(style_filename):
            style_image.save('./static/' + style_filename)
        else:
            return redirect(url_for('home'))
        return render_template('upload.html', content = content_filename, style = style_filename)

@app.route('/upload/<filename>')
def serve_image(filename):
    return send_from_directory(directory = './static/', path = filename)

@app.route('/style_transfer', methods=['GET', 'POST'])
def style_transfer():
    if request.method == 'POST':
        content_image = load_img('./static/' + content_filename)
        style_image = load_img('./static/' + style_filename)
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        image = tensor_to_image(stylized_image)
        generated_filename = 'g' + add_time('generated.jpg')
        image.save('./static/' + generated_filename)
        return render_template('model.html', output_image = generated_filename)

if (__name__ == '__main__'):
    app.run(debug = True, host = '0.0.0.0', port = 5000)