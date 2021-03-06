[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, I will train an agent to navigate and collect bananas in a large, square world by using deep neural network through Unity ML agent.

![Trained Agent][image1]

### Rule for the game
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over tested consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  Only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

3. Make sure you got python 3 and anacoda installed. To install anacoda, please refer to this link [Anacoda](https://www.anaconda.com/download/)

4. Setup anacoda deep reinforcement learning environment by create (and activate) a new environment with Python 3.6.
* **Linux** or **Mac**:
```
conda create --name drlnd python=3.6
source activate drlnd
```

* **Windows:**
```
conda create --name drlnd python=3.6 
activate drlnd
```

5. Clone the repository and install the requirements
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
6. Create and IPython Kernel for the drlnd environment and select the environment in jupyter notebook
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

7. (optional) get requirement.txt by using 
```
$pip freeze > requirements.txt.
```

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  


(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.

1. Request service limit increase in AWS for p2.xlarge instance

2. Launch an p2.xlarge instance using Deep Learning AMI with Source Code (CUDA8,Uubntu) AMI

3. Create a new security group set Protocol to TCP, port range as 8888 and SSH to the instance

4. In order to create a config file for jupyter notebook settings.
```
jupyter notebook --generate-config
```

5. Clone the repo for this projec then install the requirements
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```

6. Start the notebook
```
jupyter notebook --ip=0.0.0.0 --no-browser
```

7. Access the notebook from local browser. Access the Jupyter notebook index from your web browser by visiting: X.X.X.X:8888/?token=... (where X.X.X.X is the IP address of your EC2 instance and everything starting with :8888/?token= is what you just copied).

### Run the code

Open Navigation.ipynb in Jupyter and press Ctrl + Enter to run the code cell from first line to the last (Instruction is written in code cell)