from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_dataset, name="upload"),
    path("analysis/", views.analysis, name="analysis"),
    path("analytics/", views.analytics, name="analytics"),
    path("predict/", views.predict, name="predict"),
]
