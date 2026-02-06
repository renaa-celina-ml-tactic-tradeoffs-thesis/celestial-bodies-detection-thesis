import os
from io import BytesIO
from base64 import b64encode

import requests
import tensorflow as tf
from PIL import Image
from flask import render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, validators

from app import app
from hub.examples.image_retraining.label_image import get_labels, wiki
from hub.examples.image_retraining.reverse_image_search import reverseImageSearch

# changes for MBM
from pathlib import Path
import uuid
import pickle

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# no secret key set yet
SECRET_KEY = os.urandom(32)
app.config["SECRET_KEY"] = SECRET_KEY


class SelectImageForm(FlaskForm):
    image_url = StringField(
        "image_url",
        validators=[validators.Optional(), validators.URL()],
        render_kw={"placeholder": "Enter a URL"},
    )
    image_file = FileField(
        "file",
        validators=[
            validators.Optional(),
            FileAllowed(["jpg", "jpeg", "png", "gif"], "Invalid File"),
        ],
        render_kw={"class": "custom-file-input"},
    )

@app.route("/", methods=["GET", "POST"])
def index():
    #changes for MBM
    imageBytes = None

    form = SelectImageForm()
    if form.validate_on_submit():

        if request.files.get(form.image_file.name):
            # from file
            imageBytes = request.files[form.image_file.name].read()
            im = Image.open(BytesIO(imageBytes))
            # cant save RGBA as JPEG
            im_resize=im.convert('RGB')
            # resize using PIL
            max_width = 1000
            if(im.size[0]>max_width):
                im_resize=im.resize((max_width,int(max_width/im.size[0]*im.size[1])))
            buf = BytesIO()
            filext = request.files[form.image_file.name].filename.split('.')[-1].upper()
            if(filext=='JPG'):
                filext='JPEG'
            im_resize.save(buf, format=filext)
            # get back bytes
            imageBytes = buf.getvalue()
            print("using form")

        elif form.image_url.data:
            # from url
            response = requests.get(form.image_url.data)
            imageBytes = BytesIO(response.content).read()
            print("using url")
        else:
            # empty form
            return render_template("index.html", form=form)
        # changes for MBM
        result_uuid = create_result(imageBytes)
        return redirect(url_for("result", uuid=result_uuid))
    # print(form.errors)

    return render_template("index.html", form=form)

#changes for MBM
def create_result(imageBytes):
    cwd = os.path.join(app.root_path, "..", "hub", "examples", "image_retraining")
    try:
        celestial_object, labels = get_labels(imageBytes, cwd)
    except NameError:
        return render_template("error.html", detail="You are not supposed to be here.")
    except Exception as e:
        return render_template("error.html", detail=str(e))
    
    result_uuid = str(uuid.uuid4())
    result_file = RESULTS_DIR / f"{result_uuid}.pkl"
    result_data = {
        "celestial_object": celestial_object,
        "labels": labels,
        "imageBytes": imageBytes    
    }

    with open(result_file, "wb") as f:
        pickle.dump(result_data, f)
    return result_uuid

#changes for MBM
def load_result(uuid):
    result_path = RESULTS_DIR / f"{uuid}.pkl"
    if not result_path.exists():
        raise FileNotFoundError
    
    with open(result_path, "rb") as f:
        result_data = pickle.load(f)

    return result_data["celestial_object"], result_data["labels"], result_data["imageBytes"]


@app.route("/about")
def about():
    return render_template("about.html")

#changes for MBM
@app.get("/result")
def result():
    cwd = os.path.join(app.root_path, "..", "hub", "examples", "image_retraining")
    uuid = request.args.get("uuid")

    try:
        celestial_object, labels, imageBytes = load_result(uuid)
    except FileNotFoundError:
        return render_template("error.html", detail="404 Result not found"), 404
    title, properties, description = wiki(celestial_object, cwd)

    return render_template(
        "result.html",
        image=b64encode(imageBytes).decode("utf-8"),
        result_uuid = uuid,
        labels=labels,
        title=title,
        description=description,
        properties=properties,
    )

#changes for MBM
@app.route("/redirectToGoogle")
def redirectToGoogle():
    uuid = request.args.get("uuid")
    _, _, imageBytes = load_result(uuid)
    searchUrl = reverseImageSearch(imageBytes)
    return redirect(searchUrl, 302)