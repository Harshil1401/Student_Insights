from django.db import models

# Create your models here.
class Users(models.Model):
    First_name = models.CharField(max_length=50)
    Last_name = models.CharField(max_length=50)
    Email = models.EmailField(unique=True)
    Password = models.CharField(max_length=20)
    subscribed_to_newsletter = models.BooleanField(default=False)
    agreed_to_terms = models.BooleanField(default=False)

    def __str__(self):
        return self.Email