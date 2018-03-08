import matplotlib.pyplot as plt

#the accuraies for each beta
beta=[0.00001,0.0001,0.001,0.01,0.1,1]
result=[0.87038,0.87688,0.88249,0.88692,0.88396,0.85090]

plt.semilogx(beta, result)
plt.title('beta and accuracy relation')
plt.xlabel('beta')
plt.ylabel('accuracy')
plt.grid(True)
plt.show()
 








