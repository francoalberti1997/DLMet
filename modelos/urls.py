from django.urls import path
from .views import ModelListCreateView, ModelDetailView, IA_ModelDetailView

urlpatterns = [
    path('', ModelListCreateView.as_view(), name='blog_list_create'),
    path('<int:pk>/', ModelDetailView.as_view(), name='blog_detail'),
    path('predict/<int:pk>/', IA_ModelDetailView.as_view(), name='model-detail'),
] 