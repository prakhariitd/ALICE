from django.urls import path

from .views import results, index, thank

urlpatterns = [
    path("results/", results, name="results"),
    path("", index, name="index"),
    path("thank/", thank, name="thank"),
] 