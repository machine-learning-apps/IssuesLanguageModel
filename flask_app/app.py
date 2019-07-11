import os
import logging
import hmac
from flask import (abort, Flask, session, render_template,
                   session, redirect, url_for, request,
                   flash, jsonify)
from flask_session import Session
from urllib import request
from pathlib import Path
from inference import InferenceWrapper, pass_through
from passlib.apps import custom_app_context as pwd_context

app = Flask(__name__)

# Configure session to use filesystem. Hamel: BOILERPLATE.
app.config["SESSION_PERMANENT"] = False
Session(app)

LOG = logging.getLogger(__name__)

def init_language_model():
    """
    Downloads pre-trained language model and instantiates inference mechanism.
    """
    model_url = 'https://storage.googleapis.com/issue_label_bot/model/lang_model/models_22zkdqlr/trained_model_22zkdqlr.pkl'
    path = Path('./model_files')
    path.mkdir(exist_ok=True)

    request.urlretrieve(model_url, path/'model.pkl') 
    print('Loading model.')
    app.inference_wrapper = InferenceWrapper(model_path=path,
                                             model_file_name='model.pkl')
    print('Finished loading model.')
    

@app.route("/healthz", methods=["GET"])
def healthz():
    "route for health check."
    return jsonify({'success':True}), 200, {'ContentType':'application/json'}


# smee by default sends things to /event_handler route
@app.route("/text", methods=["POST"])
def text():
    """
    Route that allows user to send json with raw text of title and body.  This 
    route expects a payload to be sent that contains: 
    
    {'title': "some text ...", 
    'body': "some text ....}
    """
    # authenticate the request to make sure it is from a trusted party
    verify_token(request)

    # pre-process data
    title = request.json['title']
    body = request.json['body']
    data = app.inference_wrapper.process_dict({'title':title, 'body':body})
    LOG.warning(f'prediction requested for raw text: \nTITLE\n==========\n{title}\nBODY\n==========\n{body}\n')

    # make prediction
    return app.inference_wrapper.get_pooled_features(data)

@app.route("/all_issues/<string:owner>/<string:repo>", methods=["POST"])
def fetch_issues(owner, repo):
    #TODO: finish this
    """
    Retrieve the embeddings for all the issues of a repo.
    """
    installed = True
    #installed = app_installation_exists(owner=owner, repo=repo)
    if not installed:
        abort(400, description="The app is not installed on this repository.")

    if not is_public(owner, repo):
        abort(400, description="This app only works on public repositories.")

    pass


def verify_token(request):
    """Make sure request is from a trusted party."""
    # https://blog.miguelgrinberg.com/post/restful-authentication-with-flask
    token = request.headers['Token']

    if not pwd_context.verify(token, os.getenv('Token')):
        LOG.warning('Token verification failed.')
        abort(400, description="not authenticated with token.")


def is_public(owner, repo):
    "Verify repo is public."
    try:
        return requests.head(f'https://github.com/{owner}/{repo}').status_code == 200
    except:
        return False

if __name__ == "__main__":
    init_language_model()

    # make sure things reload
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0', port=os.getenv('PORT'))