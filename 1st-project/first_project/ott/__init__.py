from flask import Flask
import joblib

import config

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    app.config['MODEL'] = joblib.load(config.MODEL_PATH)
    app.config['EXPLAINER'] = joblib.load(config.EXPLAINER_PATH)

    # Blueprint
    from .views import main_views, churn_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(churn_views.bp)

    return app