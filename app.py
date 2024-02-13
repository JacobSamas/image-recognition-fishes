from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

categories = {
    0: 'fish',
    1: 'penguin'
}

# Load your trained model
model = tf.keras.models.load_model('fish_data.h5')

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', prediction="No file part")
        
        file = request.files['file']

        if file.filename == '':
            return render_template('upload.html', prediction="No selected file")
        
        if file:
            image = Image.open(io.BytesIO(file.read()))
            prepared_image = prepare_image(image, target=(224, 224))

            # Make prediction
            preds = model.predict(prepared_image)
            prediction = np.argmax(preds, axis=1) # Assuming you have class indices

            # You would need a way to map prediction indices to class names
            class_name = f"Class ID: {prediction}"

            print(class_name)

            return render_template('upload.html', prediction=class_name)

    return render_template('upload.html')

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=5000)
