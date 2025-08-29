from django.urls import path
from machine_learning import views

urlpatterns = [
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('result/', views.result_view, name='result_page'),
    path('placement_predictor/', views.placement_predictor_view, name='placement_predictor'),
    path('ctc_predictor/', views.ctc_predictor, name='ctc_predictor'),
    path("gpa_forecast/", views.gpa_forecast_view, name="gpa_forecast"),
]
