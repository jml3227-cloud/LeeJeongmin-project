from flask import Flask
import os

import config

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Blueprint
    from .views import main_views, cellsam_views, llm_views, vlm_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(cellsam_views.bp)
    app.register_blueprint(llm_views.bp)
    app.register_blueprint(vlm_views.bp)
    return app