from flask import Flask, request, jsonify,send_file
import autoencoder
import numpy as np 
from PIL import Image
import io
from matplotlib import cm
import base64


app = Flask(__name__)

@app.route("/upload_image", methods =["POST"])
def enhance_image():
	f = request.files['image'].read()

	image = Image.open(io.BytesIO(f))

	
	#print('output', type(image))

	enhanced_image = autoencoder.enhance_image(image)

	im = Image.fromarray(enhanced_image.astype('uint8'), 'RGB')

	img_byte_arr = io.BytesIO()

	im.save(img_byte_arr, format='PNG')

	img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')

	response_data = {"image": img}

	print('success')

	return jsonify(response_data)

    
    #

    #
	'''response = make_response(im)
	response.headers.set('Content-Type', 'image/jpeg')
	response.headers.set('Content-Disposition', 'attachment', filename='%s.jpg' % pid)
	return response'''
	

 	#
    #
    #

	#return jsonify({'msg': 'success'})

if __name__ == "__main__":
    app.run(debug=True)