# The code reposetory for the Master Thesis: Controlling Multiple Mobile Robots using Large Language Models
The work on this thesis is performed by Christoffer Juhl Kønig and Marcus Bøgelund Rasmussen, presented in alphabetic order based on firstname. \
Mail:\
Christoffer Juhl Kønig: chkoe19@student.sdu.dk \
Marcus Bøgelund Rasmussen: mara419@student.sdu.dk

## What code is included in this repo
### main.py
Contains the different tests, and the threads which allow the LMMs to not block the GUI, when running the full system.
### simulation.py
Contains the class Simulation, which contains the general loop in which the robots move and new tasks are started.
### ui.py
Contains the class UI, which allows the user to interact with the LLM Approach used in the simulation.
### map.py
Contains the class Map, which handles the visual repersentation in the simulation.
### robotType.py
Contains the class RobotType, which handles the different robots with-in each robot-type.
### robot.py
Contains the class Robot, which is the individual robots.
### item.py
Contains the class Item, which handles the items.

## What folders are included in this repo
### standardApproach
Contains the System Prompts for the Standard approach test.
### chainOfThought
Contains the System Prompts for the Chain-of-Thought approach test.
### ReAct
Contains the System Prompt for the ReAct approach.
### Inner Monologue
Contains the System Prompt for IM.
### modelCapabilityTests
Contains the System Prompts for the model capabilities tests.
### tasks
Contains the tasks for all tests.
### simFigures
Contains the images which are shown in the simulation.

# How to run the code
#### Step 1: Clone this repo and install the dependencies described in the requirements file.
#### Step 2: Download the Llama 2 13B quantized model used for this thesis: https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/blob/main/llama-2-13b-chat.Q5_K_M.gguf.
#### Step 3: Place the downloaded model in the folder "models"
#### Step 4: Figure out what you want to run
If you want to run the full system with GUI and simulation do **XXXXXXXX**\
To exit, press "Exit program".\\
If you want to run the full Standard Approach test do **XXXXXX**\
If you want to run the full Chain-of-Thought test do **XXXXXX**\
If you want to run the full ReAct test do **XXXXXXX**\
If you want to see a test of the Inner Monologue functionality do **xxxxxxx**\
If you want to run the full Model Capabilities test do **XXXXXXX**\
