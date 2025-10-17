from django.shortcuts import render
from rest_framework import generics
from .models import IA_Model
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

    def post(self, request, pk, format=None):        
        try:
            ia_model = IA_Model.objects.get(pk=pk)
        except IA_Model.DoesNotExist:
            return Response({'error': 'Modelo no encontrado'}, status=status.HTTP_404_NOT_FOUND)
        
        model_file = ia_model.model_file  # Suponiendo que 'instructions' contiene la ruta al archivo del modelo

        print("🔄 Preparando para cargar el modelo:", model_file)

        MODEL_PATH = os.path.join("capa_nitrurada", "modelos", f"{model_file}.h5")

        if not os.path.exists(MODEL_PATH):
            return Response({"error": f"El modelo '{model_file}.h5' no existe."}, status=404)

        print("🔄 Cargando modelo Nitride...")
        model = load_model(
            MODEL_PATH,
            custom_objects={"shape_aware_loss": shape_aware_loss, "iou_metric": iou_metric}
        )
        print("✅ Modelo cargado correctamente.\n")

        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No se envió imagen'}, status=status.HTTP_400_BAD_REQUEST)

        # Guardar imagen temporal
        input_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        with open(input_path, 'wb+') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        try:
            # 🔹 Llamar a la función única
            output_url = procesar_prediccion(input_path, model, request)
            return Response({'output_image_url': output_url})
        except Exception as e:
            print("❌ Error al procesar predicción:", e)
            return Response({'error': str(e)}, status=500)