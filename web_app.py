#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:02:10 2021

@author: bruno
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template_string, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf

def imagePreprocessing(pathImage):
    """Retorna o resultado da classificação"""
    
    filepath = "tflite_models/normaXdoente_model_quant.tflite"
    interpreter_quant = tf.lite.Interpreter(model_path=filepath)
    interpreter_quant.allocate_tensors()
    input_index = interpreter_quant.get_input_details()[0]["index"]
    output_index = interpreter_quant.get_output_details()[0]["index"]
    


    img_input = cv2.imread(pathImage)
    img_input = cv2.resize(img_input, (128, 128))
    img_input = img_input.astype('float32')/255
    
    img_input = np.array(img_input)
    img_input = img_input.reshape(1, 128, 128, 3)
    


    interpreter_quant.set_tensor(input_index, img_input)
    interpreter_quant.invoke()
    predictions = interpreter_quant.get_tensor(output_index)


    voalor_certeza = np.max(predictions)
    classe = np.argmax(predictions)
    
    return [voalor_certeza, classe]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './imgs/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


folder_temp = "arquivos/temperatura.txt"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response

@app.route('/')
def rootPage():
    return '''<!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post action='/predictFile' enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/predictFile', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result_classifica = imagePreprocessing('./imgs/' + file.filename)
            
            text_file = open(folder_temp, "w")
            text_file.write(str(result_classifica))
            text_file.close()
            
            return redirect(url_for('uploadedFile'))

    return '''<!doctype html>
    <title>Upload new File</title>
    <h1>Upload new erro File</h1>
    <form method=post action='/predictFile' enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    
    '''

@app.route('/uploadedFile', methods=['GET', 'POST'])
def uploadedFile():
    #Lendo dos valores de temperatura salvos no aquivo txt
    temp = open(folder_temp, "r")
    dados_temp1 = str(temp.read())
    temp.close()
    
    return str(dados_temp1)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
