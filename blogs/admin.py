from django.contrib import admin
from .models import Blog
from .models import Author


@admin.register(Blog)
class BlogAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'category', 'date', 'is_featured')
    search_fields = ('title', 'author', 'category')
    list_filter = ('category', 'date', 'is_featured')
    ordering = ('-date',)
    fieldsets = (
        (None, {
            'fields': ('title', 'description', 'author', 'category','body', 'date', 'read_time', 'image', 'is_featured')
        }),
    )

admin.site.register(Author)