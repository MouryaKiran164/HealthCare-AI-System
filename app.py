import json
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Model paths
MODEL_DIR = Path(__file__).parent / "models"

PNEUMONIA_PATH = MODEL_DIR / "pneumonia_mobilenetv2.keras"
LUNG_PATH = MODEL_DIR / "lung_cancer_mobilenetv2_retrained.keras"
SKIN_PATH = MODEL_DIR / "skin_cancer_mobilenetv2.keras"
SKIN_CLASSES_PATH = MODEL_DIR / "skin_cancer_class_names.json"

# Lung model class order
LUNG_CLASSES = [
    "Bengin cases",
    "Malignant cases",
    "Normal cases"
]

# Load all three models
try:
    pneumonia_model = tf.keras.models.load_model(
        PNEUMONIA_PATH, compile=False
    )

    lung_model = tf.keras.models.load_model(
        LUNG_PATH, compile=False
    )

    skin_model = tf.keras.models.load_model(
        SKIN_PATH, compile=False
    )

    with open(SKIN_CLASSES_PATH, "r", encoding="utf-8") as file:
        skin_class_names = json.load(file)

    if not isinstance(skin_class_names, list):
        raise ValueError("Skin class names must be a JSON list.")

    print("All three AI models loaded successfully.")

except Exception:
    app.logger.exception("Failed to load AI models")
    raise


def read_image(file):
    """Read an uploaded image and resize it for the model."""
    image_bytes = file.read()

    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB").resize((224, 224))

    return np.asarray(image, dtype=np.float32)


def preprocess_pneumonia_image(file):
    image_array = read_image(file)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    return np.expand_dims(image_array, axis=0)


def preprocess_lung_image(file):
    # The lung model contains its own Rescaling layer.
    image_array = read_image(file)
    return np.expand_dims(image_array, axis=0)


def preprocess_skin_image(file):
    image_array = read_image(file)

    # Match the skin model's MobileNetV2 training preprocessing.
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    return np.expand_dims(image_array, axis=0)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "HealthCare AI System",
        "status": "success",
        "message": "Backend is running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "pneumonia_model_loaded": pneumonia_model is not None,
        "lung_model_loaded": lung_model is not None,
        "skin_model_loaded": skin_model is not None
    })


@app.route("/predict", methods=["POST"])
def predict_pneumonia():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image using the image field."
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Please select an image."}), 400

    try:
        image_array = preprocess_pneumonia_image(file)
        output = pneumonia_model.predict(image_array, verbose=0)
        probability = float(np.asarray(output).reshape(-1)[0])

        prediction = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
        score = probability if probability >= 0.5 else 1 - probability

        return jsonify({
            "prediction": prediction,
            "pneumonia_probability": round(probability * 100, 2),
            "model_score": round(score * 100, 2),
            "disclaimer": (
                "Experimental prediction only; not a medical diagnosis."
            )
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "Invalid or unreadable image file."
        }), 400

    except Exception:
        app.logger.exception("Pneumonia prediction failed")
        return jsonify({
            "error": "Pneumonia analysis failed."
        }), 500


@app.route("/predict-lung", methods=["POST"])
def predict_lung():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image using the image field."
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Please select an image."}), 400

    try:
        image_array = preprocess_lung_image(file)
        output = lung_model.predict(image_array, verbose=0)
        probabilities = np.asarray(output).reshape(-1)

        if len(probabilities) != len(LUNG_CLASSES):
            raise ValueError(
                "The lung model output does not match its classes."
            )

        class_index = int(np.argmax(probabilities))
        prediction = LUNG_CLASSES[class_index]
        score = float(probabilities[class_index]) * 100

        return jsonify({
            "prediction": prediction,
            "model_score": round(score, 2),
            "class_probabilities": {
                LUNG_CLASSES[i]: round(
                    float(probabilities[i]) * 100, 2
                )
                for i in range(len(LUNG_CLASSES))
            },
            "disclaimer": (
                "Experimental image classification only. "
                "This result cannot diagnose or rule out cancer."
            )
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "Invalid or unreadable image file."
        }), 400

    except Exception:
        app.logger.exception("Lung prediction failed")
        return jsonify({
            "error": "Lung analysis failed."
        }), 500


@app.route("/predict-skin", methods=["POST"])
def predict_skin():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image using the image field."
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Please select an image."}), 400

    try:
        image_array = preprocess_skin_image(file)
        output = skin_model.predict(image_array, verbose=0)
        probabilities = np.asarray(output).reshape(-1)

        if len(probabilities) != len(skin_class_names):
            raise ValueError(
                "Skin model output does not match its class names."
            )

        class_index = int(np.argmax(probabilities))
        prediction = skin_class_names[class_index]
        score = float(probabilities[class_index]) * 100

        return jsonify({
            "prediction": prediction,
            "model_score": round(score, 2),
            "class_probabilities": {
                skin_class_names[i]: round(
                    float(probabilities[i]) * 100, 2
                )
                for i in range(len(skin_class_names))
            },
            "disclaimer": (
                "Experimental skin-lesion classification only. "
                "This result is not a medical diagnosis."
            )
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "Invalid or unreadable image file."
        }), 400

    except Exception:
        app.logger.exception("Skin prediction failed")
        return jsonify({
            "error": "Skin analysis failed."
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image too large. Maximum size is 10 MB."
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )