from django.db import models

# Create your models here.

class FruitsData(models.Model):
    fruit = models.CharField(max_length=1000)
    fruit_name = models.CharField(max_length=1000)
    in_urdu = models.CharField(max_length=1000)
    english_desc = models.CharField(max_length=1000)
    urdu_desc = models.CharField(max_length=1000)

    class Meta:
        db_table = 'fruits_data'
