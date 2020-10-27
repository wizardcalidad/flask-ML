import json
import joblib
import sklearn

import numpy as np
from flask import Flask, request

# utilities
from utils import clean_text

from flask import Flask, jsonify, request
from marshmallow import Schema, fields, ValidationError

models = {
    "bernoulli": {
        "count": joblib.load("models/bernoulli_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/bernoulli_naive_bayes_with_tfidf_vectorizer.joblib"),
    },

    "categorical": {
        "count": joblib.load("models/categorical_naive_bayes_with_count_vectorizer.joblib"),
    },
    "complement": {
        "count": joblib.load("models/complement_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/complement_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
    "gaussian": {
        "count": joblib.load("models/gaussian_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/gaussian_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
    "multinomial": {
        "count": joblib.load("models/multinomial_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/multinomial_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
}


class PredictSchema(Schema):
    model = fields.String(required=True)
    vectorizer = fields.String(required=True)
    text = fields.String(required=True)


class PredictAllSchema(Schema):
    text = fields.String(required=True)


def validate(schema_class, controller, request_data):
    # Get Request body from JSON
    schema = schema_class()

    try:
        # Validate request body against schema data types
        result = schema.load(request_data)

    except ValidationError as err:
        # Return a nice message if validation fails
        return jsonify(err.messages), 400

    # Convert request body back to JSON str
    response_data = controller(result)

    # Send data back as JSON
    return jsonify(response_data), 200


def predict(parameters: dict) -> str:
    # all the necessary parameters to select the right mode
    model = parameters.pop("model")
    vectorizer = parameters.pop("vectorizer")
    text = parameters.pop("text")

    if model == "categorical" and vectorizer == "tfidf":
        return jsonify(error="categorical does not work with tfidf vectorizer"), 400

    x = [text]  # the input
    naive_bayes_model = models[model][vectorizer]
    y = naive_bayes_model.predict(x)  # prediction

    # the final response to send back
    response = "positive" if y else "negative"
    return response


def predict_all(parameters: dict) -> dict:
    text = parameters.pop("text")

    # the final response to send back
    response = {}

    x = [text]  # the input
    for model in models:
        response[model] = {}

        for vectorizer in models[model]:
            y = models[model][vectorizer].predict(x)  # prediction
            response[model][vectorizer] = "positive" if y else "negative"

    return response


app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def predict_controller():
    return validate(PredictSchema, predict, request.json)


@app.route('/predict_all', methods=["POST"])
def predict_all_controller():
    return validate(PredictAllSchema, predict_all, request.json)


@app.route('/ping')
def ping():
    return 'pong'


if __name__ == '__main__':
    app.run()
