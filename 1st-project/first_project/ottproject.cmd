@echo off
cd C:\Flask_projects\first_project
set FLASK_APP=ott
set FLASK_DEBUG=1
C:\Python_Project\venvs\myproject\Scripts\activate
flask run --host=0.0.0.0 --port=5000