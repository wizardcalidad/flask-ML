from functools import wraps

import joblib
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


def validate_json(schema_class):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get Request body from JSON
            schema = schema_class()

            try:
                # Validate request body against schema data types
                result = schema.load(request.json)
                return f(*args, *kwargs)

            except ValidationError as err:
                # Return a nice message if validation fails
                return jsonify(err.messages), 400

        return decorated_function
    return decorator


app = Flask(__name__)


@app.route('/predict', methods=["POST"])
@validate_json(schema_class=PredictSchema)
def predict_controller():
    parameters = request.json

    # all the necessary parameters to select the right mode
    model = parameters.pop("model")
    vectorizer = parameters.pop("vectorizer")
    text = parameters.pop("text")

    if model == "categorical" and vectorizer == "tfidf":
        return jsonify(error="categorical does not  work with tfidf vectorizer"), 400

    x = [text]  # the input
    naive_bayes_model = models[model][vectorizer]
    y = naive_bayes_model.predict(x)  # prediction

    # the final response to send back
    response = "positive" if y else "negative"
    return response


@app.route('/predict_all', methods=["POST"])
@validate_json(schema_class=PredictAllSchema)
def predict_all():
    text = request.json.pop("text")

    # the final response to send back
    response = {}

    x = [text]  # the input
    for model in models:
        response[model] = {}

        for vectorizer in models[model]:
            y = models[model][vectorizer].predict(x)  # prediction
            response[model][vectorizer] = "positive" if y else "negative"

    return response


@app.route('/ping')
def ping():
    return 'pong'


if __name__ == '__main__':
    app.run()