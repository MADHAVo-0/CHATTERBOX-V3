from django.db import models

class ChatbotFeedback(models.Model):
    user_input = models.TextField()
    chatbot_response = models.TextField()
    feedback = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)