from django.db import models
from blogs.models import Author

class IA_Model(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    instructions = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)    
    image = models.URLField(blank=True, null=True)
    author = models.ForeignKey(Author, on_delete=models.SET_NULL, blank=True, null=True)
    date = models.CharField(max_length=100, blank=True, null=True)
    category = models.CharField(max_length=100, blank=True, null=True)
    model_file = models.CharField(max_length=200, blank=True, null=True)  # Nuevo campo para el archivo del modelo

    def __str__(self):
        return self.title

class Prediccion(models.Model):
    ia_model = models.ForeignKey(IA_Model, on_delete=models.CASCADE)
    input_image = models.URLField()
    output_image = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='pending')

    def __str__(self):
        return f"Predicción para {self.ia_model.title} en {self.created_at}"    