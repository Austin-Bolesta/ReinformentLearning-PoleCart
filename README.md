# ReinformentLearning-PoleCart

This if the first step in a project to use a combination of computer vision and reinforcement learning to allow and alogrithm to learn how to balance a pole-cart using openai's Gym and tensorflow. policyGradCart.py uses a [policy gradient technique](http://www.scholarpedia.org/article/Policy_gradient_methods) to learn the best series of actions (a policy) to keep a pole which is rotating around an axis at one end, standing within 15 degrees of vertical. using the motion of a cart on a 1-D track. [OpenAi's CartPole enviorment](https://gym.openai.com/envs/CartPole-v0/). 

Currently the reward function is a binary valued function for each time step. 1 for the pole standing up, and 0 for when the pole falls, or the carts hit the boundary of the screen/enviorment. Instead of this enviorment produced binary value for reward, I would like to implement a vision based layer which learns to reward itself with values over the interval [0,1] based on the actual image shown by the enviorment. This means the convolutional layer would need a similarity metric, to compare a perfectly vertical pole (reward 1), vs a pole in another position. 

