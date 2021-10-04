# USAGE
# Start the server:
# 	python -m ai.simple-keras-rest-api.run_keras_server ai/koremo/model/model_for_6.h5
# Submit a request via cURL:
# 	curl -X POST -F wav=@ai/data/F_000001.wav 'http://localhost:5000/predict'
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
from ..koremo.koremo import pred_emo
import librosa
import flask
import io
from tensorflow.compat.v1.keras.models import load_model
import sys

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure a wav was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("wav"):
			# read the wav in librosa
			wav = flask.request.files["wav"].read()
			y, sr = librosa.load(io.BytesIO(wav))

			# preprocess the wav and prepare it for classification

			# classify the input wav and then initialize the list
			# of predictions to return to the client
			results = pred_emo(model, [y])
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for (label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	model_path = sys.argv[1]
	model = load_model(model_path, compile=False)
	app.run()