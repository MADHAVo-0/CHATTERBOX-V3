import nltk
import numpy as np
import random
import string
import tensorflow as tf
import torch
import torch.nn as nn
import wikipedia

from nltk.chat.util import Chat, reflections
from django.http import JsonResponse
from django.shortcuts import render

# Initialize an empty dataset to store user interactions and feedback
user_feedback = []

# Define training data for the chatbot
training_data = [
    ['hi', ['Hello!', 'Hi there!', 'Hey!']],
    ['how are you', ['I am good, thank you.', 'I am doing well.']],
    ['what is your name', ['My name is Chatbot.', 'You can call me Chatbot.']],
    ['bye', ['Goodbye!', 'See you later.']],
    ['thanks', ['You\'re welcome!', 'No problem.']],
    ['default', ['I am still learning. Please ask me something else.']]
]

# Create a chatbot using NLTK's Chat class
chatbot = Chat(training_data, reflections)

# Define the neural network architecture
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

def chatbot_view(request):
    global chatbot
    
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        feedback = request.POST.get('feedback')
        
        # Check if the user input is a web search request
        if user_input.startswith('search:'):
            query = user_input[7:]
            try:
                page = wikipedia.page(query)
                response = page.summary
            except wikipedia.exceptions.PageError:
                response = "I'm sorry, I couldn't find any relevant information for that query."
            except wikipedia.exceptions.DisambiguationError as e:
                response = "I found multiple possible matches. Please be more specific:\n\n" + "\n".join(e.options[:5])
        else:
            response = chatbot.respond(user_input)
        
        if feedback is not None:
            if feedback.lower() == 'no':
                correct_response = request.POST.get('correct_response')
                user_feedback.append([user_input, correct_response])
                training_data.append([user_input, [correct_response]])
                chatbot = Chat(training_data, reflections)
                
                # Retrain the model with updated data
                response = chatbot.respond(user_input)
                
                # Train the neural network
                input_size = 100  # Assuming input size of 100
                hidden_size = 50
                output_size = 50
                model = ChatbotModel(input_size, hidden_size, output_size)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                for epoch in range(100):
                    inputs = torch.randn(1, input_size)
                    targets = torch.randn(1, output_size)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                # Save the trained model
                torch.save(model.state_dict(), 'chatbot_model.pth')
        
        return JsonResponse({'response': response})
    
    return render(request, 'chatbot_app/index.html')