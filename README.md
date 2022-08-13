# ALICE
Advanced Language Interpretation and Classification Environment

Usage-
To run the server, use the command-

python manage.py runserver

For the model to function, place the trained model folder in the base directory. The directiry structure should look like-

alice/classify
alice/models
...
...
alice/bert_model

The webpage HTML files can be found in the templates directory. Edit them to modify the webpage frontend.

The classify directory contains the primary backend for the server, with views.py and urls.py linking everything together.

The models directory contains the backend implementations for all modules
