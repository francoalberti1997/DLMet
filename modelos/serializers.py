from rest_framework import serializers
from .models import IA_Model

class IA_ModelSerializer(serializers.ModelSerializer):    
    class Meta:
        model = IA_Model
        fields = '__all__'
