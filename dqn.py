import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
from mxnet import ndarray as nd
from collections import deque
import matplotlib.pyplot as plt
import gym

#gym_env = gym.make('Breakout-v0')
gym_env = gym.make('Assault-v0')
#gym_env = gym.make('MsPacman-v0')

gym_env.seed(47)
np.random.seed(47)

action_space = gym_env.action_space.n
state_batch = 3

learning_rate = 0.01
epislon = 0.9           # e-greedy for random moves
gamma = 0.7             # discount factor

obs_time = 200
mb_size = 50

nb_episodes = 1000

render = True

ctx = mx.cpu(0)

class DQN(gluon.Block):
    def __init__(self):
        super(DQN, self).__init__()

        net = gluon.nn.Sequential(prefix='conv_')

        with self.name_scope():
            for i in range(4):
                net.add(gluon.nn.Conv2D((i + 1) * 18, 3, padding=(1, 1)))
                net.add(gluon.nn.BatchNorm())
                net.add(gluon.nn.Activation('relu'))
                
                net.add(gluon.nn.MaxPool2D())
            
            net.add(gluon.nn.GlobalAvgPool2D())

            net.add(gluon.nn.Dense(action_space))

        self.F = net

    def forward(self, pixels):
        sa = self.F(pixels)
        a = sa.asnumpy().argmax()
        
        return a, sa

class Enviroment(object):
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        s = self.env.reset()
        s = s.transpose((2, 0, 1))
        s = nd.array(s / 255)
        s = nd.expand_dims(s, axis=0)

        return s
    
    def render(self):
        self.env.render()

    def step(self, action):        
        s, r, done, _ = self.env.step(action)

        s = s.transpose((2, 0, 1))
        s = nd.array(s / 255)
        s = nd.expand_dims(s, axis=0)

        return s, r, done

class Agent(object):
    def __init__(self):
        self.Q = DQN()
        self.Q.initialize(ctx=ctx)
    
    def act(self, s):
        return self.Q(s)

class Playground(object):
    def __init__(self, gym_env):
        self.agent = Agent()
        self.env = Enviroment(gym_env)
        self.mse = gluon.loss.L2Loss()
        self.trainer = gluon.Trainer(self.agent.Q.collect_params(), 
                                    'adam', {'learning_rate': learning_rate})

    def train(self):
        for episode in range(nb_episodes):
            D = self.experience()
            self.learn(D)
            self.play()

    def play(self):
        done = False
        s = self.env.reset()
        count = 0

        while not done:
            a, _ = self.agent.act(s)
            s, r, done = self.env.step(a)

            count += 1

            if count % 3 == 0:
                self.env.render()

    def learn(self, D):
        mb = np.random.choice(range(obs_time), size=mb_size)
        mb = [D[m] for m in mb]
        
        for s, a, r, sn, done in mb:    
            _, t = self.agent.act(s)

            with ag.record():
                _, Q_sa = self.agent.act(sn)    

                if done:
                    t[0, int(a)] = r
                else:
                    r = r + gamma * nd.max(Q_sa)
                    t[0, int(a)] = r.asscalar()

                loss = self.mse(Q_sa, t)
                loss.backward()

            print("Loss:", loss.asscalar())

            self.trainer.step(1)

    def experience(self):
        D = deque()
        s = self.env.reset()
        
        done = False

        for i in range(obs_time):
            if np.random.rand() <= epislon:
                a = np.random.randint(0, action_space)
            else:
                a, _ = self.agent.act(s)

            sn, r, done = self.env.step(a)

            D.append((s, a, r, sn, done))

            print("Observation:", i, "Reward", r)

            if done:
                s = self.env.reset()
            else:
                s = sn
        
        return D

play = Playground(gym_env)

play.train()











