from flask import Flask, render_template, request , jsonify
import base64
import tensorflow as tf 
import cv2 
import numpy as np 


app = Flask(__name__) 
model = tf.keras.models.load_model('mnist.h5') 

model.make_predict_function() 

@app.route('/') 
def _home():
    data = 'Toe'
    return render_template('mnist.html', data = data )


@app.route('/recognize', methods = ['POST'])  
def _recognize(): 

    if request.method == 'POST': 
        data = request.get_json()
        imageBase64 = data['image']
        imgBytes = base64.b64decode(imageBase64)

        with open("temp.jpg", "wb" ) as temp : 
            temp.write(imgBytes)

        image = cv2.imread('temp.jpg') 
        image = cv2.resize(image,(28,28), interpolation=cv2.INTER_AREA)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image_prediction=np.reshape(image_gray, (28,28,1))
        image_prediction=(255-image_prediction.astype('float')) / 255 

        prediction = np.argmax(model.predict(np.array([image_prediction])), axis=-1) 



        return jsonify({'prediction': str(prediction[0]),
                        'status': True 
                        })

if __name__ == '__main__' : 
    app.run(debug=True)



