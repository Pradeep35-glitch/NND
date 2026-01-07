import math

free = 1
win = 0
offer = 1

w_free_h1 = 0.5
w_win_h1 = -0.2
w_offer_h1 = 0.3

w_free_h2 = 0.4
w_win_h2 = 0.1
w_offer_h2 = -0.5

w_h1_out = 0.7
w_h2_out = 0.2

def sigmoid(x):
return 1 / (1 + math.exp(-x))

h1 = free*w_free_h1 + win*w_win_h1 + offer*w_offer_h1
h2 = free*w_free_h2 + win*w_win_h2 + offer*w_offer_h2

print("H1 value:", h1)
print("H2 value:", h2)

spam_input = h1*w_h1_out + h2*w_h2_out
spam_probability = sigmoid(spam_input)

print("Spam input:", spam_input)
print("Spam probability:", spam_probability)

