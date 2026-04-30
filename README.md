# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Mario Viofer J

### Register Number: 212223100032

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
X=torch.linspace(1,50,50).reshape(-1,1)
torch.manual_seed(71)
e=torch.randint(-8,9,(50,1),dtype=torch.float)
Y=2*X+1+e
plt.scatter(X.numpy(),Y.numpy(),color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Data For Linear Regression')
plt.show()
class Model(nn.Module):
  def __init__(self,in_features,out_features):
    super().__init__()
    self.linear=nn.Linear(in_features,out_features)

  def forward(self,x):
    return self.linear(x)
torch.manual_seed(59)
model=Model(1,1)
print(model)
print('Weight:',model.linear.weight.item())
print('Bias:',model.linear.bias.item())
loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.0001)
epochs=50
losses=[]
for epoch in range(1,epochs+1):
  optimizer.zero_grad()
  y_pred=model(X)
  loss=loss_function(y_pred,Y)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()
  print(f"epoch: {epoch:2} loss: {loss.item():10.08f}  weight: {model.linear.weight.item():10.08f}  bias: {model.linear.bias.item():10.08f}")

plt.plot(range(epochs),losses)
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.show()
final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()
print(f"\nFinal Weight : {final_weight:.8f} ,Final Bias: {final_bias}")
x1=torch.tensor([X.min().item(),X.max().item()])
y1=x1*final_weight+final_bias
plt.scatter(X,Y,label='Original Data')
plt.plot(x1,y1,'r',label='Best-Fit Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()
x_new=torch.tensor([[120.0]])
y_new_pred=model(x_new).item()
print(f"Prediction for x = 120 : {y_new_pred:.8f}")

```

### Dataset Information
Include screenshot of the generated data
<img width="570" height="449" alt="image" src="https://github.com/user-attachments/assets/7c34c861-d80e-4eea-ae0a-e3b64b4d6ae7" />


### OUTPUT
Training Loss Vs Iteration Plot
<img width="577" height="429" alt="image" src="https://github.com/user-attachments/assets/59d2d3de-3709-4089-85af-99b97059d311" />

Best Fit line plot
<img width="569" height="448" alt="image" src="https://github.com/user-attachments/assets/b8c19212-92d7-427c-97cd-7fa83c318639" />

### New Sample Data Prediction
Include your sample input and output here
<img width="410" height="685" alt="image" src="https://github.com/user-attachments/assets/3ff01e8d-110c-4f7a-b335-f2991d52e8b8" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
