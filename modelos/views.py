from django.shortcuts import render
from rest_framework import generics
from .models import IA_Model, Prediccion
from .serializers import IA_ModelSerializer
import os
from django.conf import settings
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from capa_nitrurada.testing_model import procesar_prediccion
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from capa_nitrurada.testing_model import shape_aware_loss, iou_metric
import tensorflow as tf
import threading
import os

# global para cachear modelos
loaded_models = {}

# Lista todos los blogs o crea uno nuevo
class ModelListCreateView(generics.ListCreateAPIView):
    queryset = IA_Model.objects.all()
    serializer_class = IA_ModelSerializer

# Devuelve, actualiza o borra un IA_Model específico
class ModelDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = IA_Model.objects.all()
    serializer_class = IA_ModelSerializer

class IA_ModelDetailView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request, pk, format=None):
        try:
            model = IA_Model.objects.get(pk=pk)
        except IA_Model.DoesNotExist:
            return Response({'error': 'Modelo no encontrado'}, status=status.HTTP_404_NOT_FOUND)
        serializer = IA_ModelSerializer(model)
        return Response(serializer.data)

    def get_model(self, model_file):
        MODEL_PATH = os.path.join("capa_nitrurada", "modelos", f"{model_file}.h5")
        if not os.path.exists(MODEL_PATH):
            return None

        # Solo aplicar memory growth si hay GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print("⚠️ No se pudo setear memory growth:", e)

        model = load_model(
            MODEL_PATH,
            custom_objects={"shape_aware_loss": shape_aware_loss, "iou_metric": iou_metric}
        )
        return model

    def post(self, request, pk, format=None):
        try:
            ia_model = IA_Model.objects.get(pk=pk)
        except IA_Model.DoesNotExist:
            return Response({'error': 'Modelo no encontrado'}, status=status.HTTP_404_NOT_FOUND)
        
        model_file = ia_model.model_file
        model = self.get_model(model_file)
        if model is None:
            return Response({"error": f"El modelo '{model_file}.h5' no existe."}, status=404)

        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No se envió imagen'}, status=status.HTTP_400_BAD_REQUEST)

        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        input_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
        with open(input_path, 'wb+') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        input_url = settings.MEDIA_URL + image_file.name

        # Crear objeto Prediccion en estado pendiente
        prediccion = Prediccion.objects.create(
            ia_model=ia_model,
            input_image=input_url,
            status='pending'
        )

        # Iniciar procesamiento en segundo plano
        threading.Thread(target=self._procesar_prediccion_async, args=(prediccion.id, model, input_path, request)).start()

        # Responder de inmediato con ID
        return Response({
            'message': 'Predicción en proceso',
            'prediccion_id': prediccion.id,
            'input_image_url': input_url
        }, status=200)

    def _procesar_prediccion_async(self, prediccion_id, model, input_path, request):
        from django.db import transaction
        try:
            pred = Prediccion.objects.get(id=prediccion_id)
            output_url = procesar_prediccion(input_path, model, request)
            if output_url.startswith('http://'):
                output_url = output_url.replace('http://', 'https://', 1)
            
            with transaction.atomic():
                pred.output_image = output_url
                pred.status = 'done'
                pred.save()
        except Exception as e:
            print("❌ Error procesando predicción:", e)
            pred = Prediccion.objects.get(id=prediccion_id)
            pred.status = 'error'
            pred.save()

class PrediccionStatusView(APIView):
    def get(self, request, prediccion_id):
        try:
            pred = Prediccion.objects.get(id=prediccion_id)
        except Prediccion.DoesNotExist:
            return Response({'error': 'Predicción no encontrada'}, status=404)

        return Response({
            'id': pred.id,
            'status': pred.status,
            'input_image': pred.input_image,
            'output_image': pred.output_image
        })
