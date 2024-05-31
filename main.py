from llama_cpp import Llama
import csv
from tqdm import tqdm
import json
import threading
import time
import random
import simulation

modelPath = "model/llama-2-13b-chat.Q5_K_M.gguf"
def returnModelPath():
  return modelPath

def readPromptsFromTxt(name):
  file = open(name, 'r')
  lines = file.readlines()
  for i in range(len(lines)):
    if lines[i][-1:] == "\n":
     lines[i] = lines[i][:-1]
  return lines

def append_user(message, message_history):
  message_history.append({"role": "user", "content":message})
  return message_history

def append_assistant(message, message_history):
  message_history.append({"role": "assistant", "content":message})
  return message_history

def append_system(message, message_history):
  message_history.append({"role": "system", "content":message})
  return message_history

def create_response(llm, message_history):
  response=llm.create_chat_completion(messages=message_history,max_tokens=1000, temperature=0.0, top_p=1.0,
                  repeat_penalty=1.1, top_k=1)
  reply = response["choices"][0]["message"]["content"]

  message_history = append_assistant(reply, message_history)
  return message_history, reply

def getOverviewReAct(task):
  llmOverview = Llama(
  model_path=modelPath,
  seed = 1,
  n_threads=4, # CPU cores
  n_ctx = 4096, # Allows for larger message history
  n_batch=2048, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
  n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool.
  verbose=False # True for messages from Llama)
  )
  
  llmOverview.reset()
  messagesExplain = [
    {
      "role": "system",
      "content" :"You are a robot task planner and when a thought is provided to you, you must explain which action is needed to solve the task. You can choose one of the following actions: scoop up multiple objects, manipulate light object, manipulate heavy object, and move without manipulating. You can ONLY choose ONE of the actions so please choose wisely. When an action is chosen you MUST end your explanation saying exactly: \"Consider which function could be used to obtain information about the robots in the environment.\"\nThe following are examples of how a robot task planner could explain a task using the possible actions. First example: The task is: Move branches from 3,1 to 12,5. EXPLANATION: The task is to move branches from 3,1 to 12,5. For this task the action scoop up multiple objects could be used, since multiple branches needs to be moved. \nConsider which function could be used to obtain information about the robots in the environment.\nSecond example: The task is: Grab a flask from the kitchen table located at 10,10 and move to 1,7 after. EXPLANATION: The task is to grab a flask from the kitchen table at 10,10 and move to 1,7 after. For this task the action manipulate light object could be used, since a glass is a light object and it needs to be manipulated.\nConsider which function could be used to obtain information about the robots in the environment. \nThird example: The task is: Move a large stone from 0,12 to 65,8. EXPLANATION: The task is to move a large stone from 0,12 to 65,8. For this task the action manipulate heavy object could be used, since a large stone is a heavy object and it needs to be manipulated.\nConsider which function could be used to obtain information about the robots in the environment. \nFourth example: The task is: Move from 89,7 to 1,1. EXPLANATION: The task is to move from 89,7 to 1,1. For this task the action move without manipulating could be used, since the task does not involve manipulating any objects, just moving from 89,7 to 1,1.\nConsider which function could be used to obtain information about the robots in the environment.\nFifth example: The task is: Move from 0,0 to 7,3 three times. EXPLANATION: Move from 0,0 to 7,3 three times. For this task the action move without manipulating could be used, since the task does not involve manipulating any objects, just moving from 0,0 to 7,3 three times.\nConsider which function could be used to obtain information about the robots in the environment.\nSixth example: The task is: Transport a tree from 1,9 to 12,5. EXPLANATION: Transport a tree from 1,9 to 12,5. For this task the action move without manipulating could be used, since the task does not involve manipulating any objects, just transporting an object from 1,9 to 12,5.\nConsider which function could be used to obtain information about the robots in the environment.\n"

    },
    {
      "role": "user",
      "content": "The task is:" + task
    },
  ]
  
  first = True
  stringOut = ""
  for out in llmOverview.create_chat_completion(messages=messagesExplain, max_tokens=1000, temperature=0.0, top_p=1.0, repeat_penalty=1.1, top_k=1,stream=True):
    if first is True:
      first = False
    elif out["choices"][0]["finish_reason"] == "stop":
      print("\n")
      break
    else:
      stringOut += out["choices"][0]["delta"]["content"]
      print(out["choices"][0]["delta"]["content"],sep='',end="")
  return stringOut
  
def extract_coordinates(llmObject,task ):

  llmObject.reset()
  messagesCoordinates = [
    {
      "role": "system",
      "content": "Your job is to extract coordinates from the task given by the user. The task start with the keyword TASK:. If a task should be repeated once follow this format: EXTRACTED_COORDINATE(S): (x,y). If a task should be repeated n times follow this format: EXTRACTED_COORDINATE(S): {(x,y),(x,y)}_1, {(x,y),(x,y)}_2, ..., {(x,y),(x,y)}_n. You cannot ask any questions, just extract coordinates from the task, even if the coordinates are repersented by letters. Here are some examples: TASK: Move from 31,10 to 5,1. EXTRACTED_COORDINATE(S): (31,10), (5,1). TASK: Move to 1,1 from 13,2. EXTRACTED_COORDINATE(S): (13,2), (1,1). TASK: Move from 12,6 to 1,90 four times. EXTRACTED_COORDINATE(S): {(12,6), (1,90)}, {(12,6), (1,90)}, {(12,6), (1,90)}, {(12,6), (1,90)}. TASK: Move from 5,9 to 123,9 three times. EXTRACTED_COORDINATE(S): {(5,9), (123,9)}, {(5,9), (123,9)}, {(5,9), (123,9)}. TASK: Move from 11,71 to 13,71 7 times. EXTRACTED_COORDINATE(S): {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}. TASK: Move from 11,71 to 13,71 8 times. EXTRACTED_COORDINATE(S): {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}, {(11,71), (13,71)}. TASK: Move from 1,5 to 31,9 3 times. EXTRACTED_COORDINATE(S): {(1,5), (31,9)}, {(1,5), (31,9)}, {(1,5), (31,9)}. TASK: Move from 2,16 to 13,2 but do something at 93,6 first. EXTRACTED_COORDINATE(S): (93,6), (2,16), (13,2). TASK: Move from 8,23 to 4,1 but go to 4,1 first. EXTRACTED_COORDINATE(S): (4,1), (8,23), (4,1). TASK: Get the ball from the chair. The chair is located at 0,1. EXTRACTED_COORDINATE(S): (0,1). TASK: Get the ball from the floor and move it to the table. The floor is located at 0,10 and the table is located at 17,8. EXTRACTED_COORDINATE(S): (0,10), (17,8). TASK: Get the ball from the shelf located at 5,1 and move it 89,1, but move through 3,1 when you have the ball. EXTRACTED_COORDINATE(S): (5,1), (3,1), (89,1).  TASK: Move from coordinate C to coordinate D. EXTRACTED_COORDINATE(S): (C), (D). TASK: Move from 35,1 to 27,75 to 61,102. Repeat this three times. EXTRACTED_COORDINATE(S): {(35,1), (27,75), (61,102)}, {(35,1), (27,75), (61,102)}, {(35,1), (27,75), (61,102)}. TASK: Move from 2,2 to 4,4 to 2,2. Repeat this three times. EXTRACTED_COORDINATE(S): {(2,2), (4,4), (2,2)}, {(2,2), (4,4), (2,2)}, {(2,2), (4,4), (2,2)}. TASK: Move from 2,4 to 51,2 to 2,4. Repeat this two times. EXTRACTED_COORDINATE(S): {(2,4), (51,2), (2,4)}, {(2,4), (51,2), (2,4)}. TASK: Move from 10,10 to 12,12, but clear 12,12 from snow first. EXTRACTED_COORDINATE(S): (12,12), (10,10), (12,12). TASK: Move from 12,34 to 43,21, but clear 12,34 from leaves first. EXTRACTED_COORDINATE(S): (12,34), (43,21), (12,34). TASK: Move from coordinate B to coordinate C. EXTRACTED_COORDINATE(S): (B), (C). TASK: Move from 4,90 to 123,321 to 4,90. Repeat this 3 times. EXTRACTED_COORDINATE(S): {(4,90), (123,321), (4,90)}, {(4,90), (123,321), (4,90)}, {(4,90), (123,321), (4,90)}. TASK: Move from 2,3 to 3,2 to 2,3. Repeat this four times. EXTRACTED_COORDINATE(S): {(2,3), (3,2), (2,3)}, {(2,3), (3,2), (2,3)},{(2,3), (3,2), (2,3)},{(2,3), (3,2), (2,3)}. TASK: Move from 9,10 to 0,12, but clear 0,12 from a pile of gravel first. EXTRACTED_COORDINATE(S): (0,12), (9,10), (0,12). TASK: Move from C to D. EXTRACTED_COORDINATE(S): (C), (D). The next thing provided to you will be the task you have to extract coordinates from."
    },
    {
      "role": "user",
      "content": "TASK: " + task
    }
  ]

  first = True
  stringOut = ""
  for out in llmObject.create_chat_completion(messages=messagesCoordinates, max_tokens=1000, temperature=0.0, top_p=1.0, repeat_penalty=1.1, top_k=1,stream=True):
    if first is True:
      first = False
    elif out["choices"][0]["finish_reason"] == "stop":
      print("\n")
      break
    else:
      stringOut += out["choices"][0]["delta"]["content"]
      print(out["choices"][0]["delta"]["content"],sep='',end="")
  return stringOut
  
def finishCoordinates(extractedString):
    out = extractedString[extractedString.rindex(":")+1:]
    if out[0] == " ":
        out = out[1:]
    if out[-1] == ".":
        out = out[:-1]
    return out  
  
def overviewIM(task,robotPos):
  llmOverview = Llama(
  model_path=modelPath,
  seed = 1,
  n_threads=10, # CPU cores
  n_ctx = 1024, # Allows for larger message history
  n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
  n_gpu_layers=-1, # Change this value based on your model and your GPU VRAM pool.
  verbose=False # True for messages from Llama)
  )
  systemPrompt=readPromptsFromTxt("Inner Monologue/IMoverview.txt")[0]
  
  llmOverview.reset()
  messagesOverviewIM = [
    {
      "role": "system",
      "content": systemPrompt
    },
    {
      "role": "user",
      "content": "The task is:" + task + " Robot position: " + str(robotPos) +"."
    },
  ]
  
  first = True
  stringOut = ""
  for out in llmOverview.create_chat_completion(messages=messagesOverviewIM, max_tokens=1000, temperature=0.0, top_p=1.0, repeat_penalty=1.1, top_k=1,stream=True):
    if first is True:
      first = False
    elif out["choices"][0]["finish_reason"] == "stop":
      print("\n")
      break
    else:
      stringOut += out["choices"][0]["delta"]["content"]
      print(out["choices"][0]["delta"]["content"],sep='',end="")
  return stringOut

def succesDetector(succesChance=0.5):
  current = random.uniform(0,1)
  if current <= succesChance:
    return True
  return False

def modelCapabilitiesTest():
  pathToTest = "tasks/wordingTest.txt"
  pathToSystemPrompt = "modelCapabilityTests/wordingSystemPrompt.txt"
  outputName = "wording_Test"
  modelCapabilitiesTestGeneric(pathToTest=pathToTest,pathToSystemPrompt=pathToSystemPrompt,outputName=outputName)
  pathToTest = "tasks/positionalInformationTest.txt"
  pathToSystemPrompt = "modelCapabilityTests/positionalInformationSystemPrompt.txt"
  outputName = "positional_Information_Test"
  modelCapabilitiesTestGeneric(pathToTest=pathToTest,pathToSystemPrompt=pathToSystemPrompt,outputName=outputName)
  pathToTest = "tasks/combinedTest.txt"
  pathToSystemPrompt = "modelCapabilityTests/combinedSystemPrompt.txt"
  outputName = "combined_Test"
  modelCapabilitiesTestGeneric(pathToTest=pathToTest,pathToSystemPrompt=pathToSystemPrompt,outputName=outputName)

def modelCapabilitiesTestGeneric(pathToTest,pathToSystemPrompt,outputName):
  llm = Llama(
    model_path=modelPath,
    seed = 1,
    n_threads=24, # CPU cores
    n_ctx = 4096, # Allows for larger message history
    n_batch=2048, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool.
    verbose=False # True for messages from Llama
    )
  tasks = readPromptsFromTxt(pathToTest)
  print(len(tasks))
  startup = readPromptsFromTxt(pathToSystemPrompt)
  print(len(startup))
  for i in range(len(startup)):
    name = "modelCapabilityTests/results/"+outputName +str(i)+ ".csv"
    with open(name, 'w', encoding='UTF8', newline='') as f:
      llmSystemPrompt = startup[i]
      writer = csv.writer(f)
      writer.writerow(["Startup first:", llmSystemPrompt, "Wrong", modelPath])
      writer.writerow(["Promt","Response"])
      print(llmSystemPrompt)

    for task in tqdm(tasks):
      #Resets the llms after each itteration
      llm.reset()
      
      #Gives the resetted llm the startup promt and resetts the promt history
      history = append_system(llmSystemPrompt, [])
      #second_history = append_system(secondaryLLM, [])
      
      #Gives the promt to the llm and asks for the explanation afterwards
      history = append_user("TASK: " + task, history)
      #history, reply = create_response(llm, history)
      reply = "ja"
    
      
      with open(name, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        #writer.writerow(["Promt","Succes", "Did not understand task", "Points not included","Which LLM Fucked up (0=none, 3=both)", "Reply","Reply from other LLM"])
        writer.writerow([task,reply])
      
      print("-------------------------------------------------------------------------------------------------")
      #print("Primary:", i+1, "Secondary:", j+1)
      print("Task:", task)
      print("-------------------------------------------------------------------------------------------------")
      print(reply)
      print("-------------------------------------------------------------------------------------------------")

tools = [
    {
    "type": "function",
    "function": {
      "name": "finish",
      "description": "Returns the final answer.",
      "parameters": {
          "type": "object",
          "properties": {
            "selected_robot": {
                "type": "string",
                "description": "Includes which robot number is selected.",
              },
            "EXTRACTED_COORDINATE(S)": {
                "type": "string",
                "description": "Includes ALL the task coordinates.",
              },
          },
          "required": ["robot","EXTRACTED_COORDINATE(S)"],
      },
    },   
  },
  {
    "type": "function",
    "function": {
      "name": "get_overview_of_task",
      "description": "Based on the task, provide an overview of the task and all of the necessary steps to complete the task.",
      "parameters": {
          "type": "object",
          "properties": {
            "overview_of_task": {
                "type": "string",
                "description": "The full overview of the task presented as a string."
              },
          },
          "required": ["topic","task"],
      },
    },   
  },
  {
    "type": "function",
    "function": {
      "name": "extract_coordinates",
      "description": "Based on the task provide the coordinates necessary to complete the task.",
      "parameters": {
          "type": "object",
          "properties": {
            "EXTRACTED_COORDINATE(S)": {
                "type": "string",
                "description": "The necessary coordinates to complete the task."
              },
          },
          "required": ["task"],
      },
    },   
  },
  {
    "type": "function",
    "function": {
      "name": "think",
      "description": "Reasons about the task and all the information available.",
      "parameters": {
          "type": "object",
          "properties": {
              "thought": {
                  "type": "string",
                  "description" : "Reason about the task and all the information available.",
              },
          },
          "required": ["task"],
      },
    },   
  },
  {
    "type": "function",
    "function": {
      "name": "describe_robots",
      "description": "Returns information about all of the robots in the environment. It is a very good idea to use this function to make sure that the correct robot is chosen. describe_robots is not needed if the task mention a specific robot type.",
      "parameters": {
          "type": "object",
          "properties": {
              "description": {
                  "type": "string",
                  "description": "Information about all of the robots in the environment.",
              },
          },
          "required": ["task"],
      },
    },   
  },
  {
    "type": "function",
    "function": {
      "name": "select_robot",
      "description": "Use this function to select the most suitable robot for the task based on the available information.",
      "parameters": {
          "type": "object",
          "properties": {
            "selected_robot": {
              "type": "string",
              "description": "The most suitable robot type.",
            },
          },
          "required": ["topic","task"],
      },
    },   
  }, 
]

toolsConversation = [
    {
    "type": "function",
    "function": {
      "name": "think",
      "description": "Think about the given question and consider which function could be used next.",
      "parameters": {
          "type": "object",
          "properties": {
              "thought": {
                  "type": "string",
                  "description" : "Think about the given question and consider which function could be used next.",
              },
          },
          "required": ["question"],
      },
    },   
    },
  {
  "type": "function",
  "function": {
    "name": "get_status_of_tasks",
    "description": "Use this function to get the status for all tasks.",
    "parameters": {
        "type": "object",
        "properties": {
            "status_of_tasks": {
                "type": "string",
                "description" : "The status of all tasks",
            },
        },
        "required": ["status of tasks"],
    },
  },   
},
  {
  "type": "function",
  "function": {
    "name": "get_status_of_robots",
    "description": "Use this function to get the status of each robot.",
    "parameters": {
        "type": "object",
        "properties": {
            "status_of_robots": {
                "type": "string",
                "description" : "The status of all the robots.",
            },
        },
        "required": ["task"],
    },
  },   
},
    {
  "type": "function",
  "function": {
    "name": "get_position_of_robots",
    "description": "Use this function to the position of all robots.",
    "parameters": {
        "type": "object",
        "properties": {
            "robot_positions": {
                "type": "string",
                "description" : "The position of all robots.",
            },
        },
        "required": ["task"],
    },
  },   
},
]

toolsIM = [
  {
  "type": "function",
  "function": {
    "name": "move_robot",
    "description": "Use this function if the robot has to move to a new location.", #to move the robot to the next coordinate.",
    "parameters": {
        "type": "object",
        "properties": {
        },
    },
  },   
},
  {
  "type": "function",
  "function": {
    "name": "pick_up_object",
    "description": "Use this function to pick up an object.",
    "parameters": {
        "type": "object",
        "properties": {
            "object": {
                "type": "string",
                "description" : "The object which has been picked up.",
            },
        },
        "required": ["object"],
    },
  },   
},
    {
  "type": "function",
  "function": {
    "name": "place_object",
    "description": "Use this function to place an object.",
    "parameters": {
        "type": "object",
        "properties": {
            "object": {
                "type": "string",
                "description" : "The object which has been placed.",
            },
        },
        "required": ["object"],
    },
  },   
},

  {
    "type": "function",
    "function": {
      "name": "task_completed",
      "description": "When you think the robot is done with the robotic task.",
      "parameters": {
          "type": "object",
          "properties": {
            "isTaskComplete": {
                "type": "string",
                "description": "Should say \"Yes\" if the task is complete.",
              },
          },
      },
    },   
  },
  {
    "type": "function",
    "function": {
      "name": "think",
      "description": "Reason about all the information available especially consider the robot's position.",
      "parameters": {
          "type": "object",
          "properties": {
              "thought": {
                  "type": "string",
                  "description" : "Reason about the task and all the information available.",
              },
          },
          "required": ["task"],
      },
    },   
  },
]      
      
def standardApproachTest():
  llm = Llama(
    model_path=modelPath,
    seed = 1,
    n_threads=24, # CPU cores
    n_ctx = 4096, # Allows for larger message history
    n_batch=2048, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool.
    verbose=False # True for messages from Llama
    )
  
  
  tasks = readPromptsFromTxt("tasks/tasks19Jan.txt")
  environmentInfo = readPromptsFromTxt("standardApproach/environmentInfo.txt")
  print("environmentInfo:", len(environmentInfo))
  robotInfo = readPromptsFromTxt("standardApproach/robotInfo.txt")
  print("robotInfo:", len(robotInfo))
  rules = readPromptsFromTxt("standardApproach/rules.txt")
  print("rules:", len(rules))
  examples = readPromptsFromTxt("standardApproach/examples.txt")
  print("examples:", len(examples))
  for en in range(len(environmentInfo)):
    for ro in range(len(robotInfo)):
      for ru in range(len(rules)):
        for ex in range(len(examples)):
          current = "en" +str(en+1)+"ro"+str(ro+1)+"ru" + str(ru+1)+"ex"+str(ex+1)
          name = "standardApproach/results/" + current + ".csv"
          with open(name, 'w', encoding='UTF8', newline='') as f:
            systemPrompt = environmentInfo[en] + robotInfo[ro] + rules[ru] + examples[ex]
            writer = csv.writer(f)
            writer.writerow(["Startup Message:", current + "\n" + systemPrompt, "Length of startup message:", len(systemPrompt), modelPath])
            writer.writerow(["Prompt","Succes", "Did not understand task", "Points not included","Reply"])
            print(systemPrompt)
          for task in tqdm(tasks):
            #Resets the llms after each itteration
            llm.reset()
            
            
            #Gives the resetted llm the startup promt and resetts the promt history
            message_history = append_system(systemPrompt, [])
            
            #Gives the promt to the llm and asks for the explanation afterwards
            message_history = append_user("TASK: " + task, message_history)
            message_history, reply = create_response(llm, message_history)
            
            with open(name, 'a', encoding='UTF8', newline='') as f:
              writer = csv.writer(f)
              writer.writerow([task,"", "", "", reply])
            
            print("-------------------------------------------------------------------------------------------------")
            print("Task:", task)
            print("-------------------------------------------------------------------------------------------------")
            print(reply)
            print("-------------------------------------------------------------------------------------------------")

def chainOfThoughtApproachTest():
  llm1 = Llama(
    model_path=modelPath,
    seed = 1,
    n_threads=24, # CPU cores
    n_ctx = 4096, # Allows for larger message history
    n_batch=2048, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool.
    verbose=False # True for messages from Llama
    )
  llm2 = Llama(
    model_path=modelPath,
    seed = 1,
    n_threads=24, # CPU cores
    n_ctx = 4096,
    n_batch=2048, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool.
    verbose=False # True for messages from Llama
    )
  
  #LLM1 system prompt
  environmentInfo1 = readPromptsFromTxt("chainOfThought/1_environment_information.txt")
  print("environmentInfo1:", len(environmentInfo1))
  robotInfo1 = readPromptsFromTxt("chainOfThought/1_robot_information.txt")
  print("robotInfo1:", len(robotInfo1))
  rules1 = readPromptsFromTxt("chainOfThought/1_rules.txt")
  print("rules1:", len(rules1))
  examples1 = readPromptsFromTxt("chainOfThought/1_examples.txt")
  print("examples1:", len(examples1))

  #LLM2 system prompt
  robotInfo2 = readPromptsFromTxt("chainOfThought/2_robot_information.txt")
  print("robotInfo2:", len(robotInfo2))
  rules2 = readPromptsFromTxt("chainOfThought/2_rules.txt")
  print("rules2:", len(rules2))
  examples2 = readPromptsFromTxt("chainOfThought/2_examples.txt")
  print("examples2:", len(examples2))


  tasks = readPromptsFromTxt("tasks/tasks19Jan.txt")
  print("Tasks:", len(tasks))
  print(tasks)

  for en1 in range(len(environmentInfo1)):
    for ro1 in range(len(robotInfo1)):
      for ru1 in range(len(rules1)):
        for ex1 in range(len(examples1)):
          for ro2 in range(len(robotInfo2)):
            for ru2 in range(len(rules2)):
              for ex2 in range(len(examples2)):
                current = "en1-" +str(en1+1)+"_ro1-"+str(ro1+1)+"_ru1-" + str(ru1+1)+"_ex1-"+str(ex1+1)+"_ro2-"+str(ro2+1) + "_ru2-" + str(ru2+1) + "_ex2-" + str(ex2+1)
                name = "chainOfThought/results/" + current + ".csv"
                with open(name, 'w', encoding='UTF8', newline='') as f:
                  llm1SystemPrompt = environmentInfo1[en1] + robotInfo1[ro1] + rules1[ru1] + examples1[ex1]
                  llm2SystemPrompt = robotInfo2[ro2] + rules2[ru2] + examples2[ex2]
                  writer = csv.writer(f)
                  writer.writerow(["Startup first:", current + "\n" + llm1SystemPrompt, "Length of first message:", len(llm1SystemPrompt), "Startup second:", llm2SystemPrompt, "Length of secondary message:", len(llm2SystemPrompt), "combined length:", len(llm1SystemPrompt)+len(llm2SystemPrompt), modelPath])
                  writer.writerow(["Prompt","Succes", "Did not understand task", "Points not included","Which LLM Fucked up (0=none, 3=both)", "Reply","Reply from other LLM"])
                for task in tqdm(tasks):
                  #Resets the llms after each itteration
                  llm1.reset()
                  llm2.reset()
                  
                  #Gives the resetted llm the startup prompt and resetts the prompt history
                  first_history = append_system(llm1SystemPrompt, [])
                  second_history = append_system(llm2SystemPrompt, [])
                  
                  #Gives the prompt to the llm and asks for the explanation afterwards
                  first_history = append_user("TASK: " + task, first_history)
                  first_history, first_reply = create_response(llm1, first_history)
                  
                  
                  #Uses the first llm to create explanations which are feeded to the second llm
                  second_history = append_user("TASK: " + task + " " + first_reply, second_history)
                  second_history, second_reply = create_response(llm2, second_history)
                  
                  with open(name, 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Task","Succes", "Did not understand task", "Points not included","Which LLM Fucked up (0=none, 3=both)", "Reply","Reply from other LLM"])
                    writer.writerow([task,"", "", "", "0", first_reply, second_reply])
                  
                  print("-------------------------------------------------------------------------------------------------")
                  print("Task:", task)
                  print("-------------------------------------------------------------------------------------------------")
                  print(first_reply)
                  print("-------------------------------------------------------------------------------------------------")
                  print(second_reply)
                  print("-------------------------------------------------------------------------------------------------")

def reActApproachTest():
  llm = Llama(
      model_path=modelPath,
      chat_format="chatml-function-calling",
      seed = 1,
      n_threads=10, # CPU cores
      n_ctx = 4*4096, # Allows for larger message history
      n_batch= 4*2048, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
      n_gpu_layers=-1, # Change this value based on your model and your GPU VRAM pool.
      verbose=False # True for messages from Llama)
    )  

  tasks = readPromptsFromTxt("tasks/tasks6April.txt")
  print("Tasks:", len(tasks))
  print(tasks)

  systemPrompt = readPromptsFromTxt("ReAct/reActSystemPrompt.txt")

  name = "ReAct/Results/ReActTest.csv"
  with open(name, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Startup Message:", systemPrompt])
    writer.writerow(["Prompt","Succes", "Outputs", "Total Tokens"])
  for task in tqdm(tasks):
    #Resets the llms after each itteration
    llm.reset()
    messages = [
    {
      "role": "system",
      "content": systemPrompt
    },
    {
      "role": "user",
      "content": "Task: " + task + "\nThought 1: " 
    }
    ]
    print("\nCurrent Task:", task)  
    total_tokens = 0
    i = 0
    stuck_counter = 0
    coordinates = None
    while True:
      # -------------------------- Think --------------------------
      think = llm.create_chat_completion(messages=messages, max_tokens=1000, temperature=0.0, top_p=1.0,
                        repeat_penalty=1.1, top_k=1, tools=tools,tool_choice={"type": "function", "function": {"name": "think"}})
      

      total_tokens = think["usage"]["total_tokens"]
      if(total_tokens > 15000):
        print("totalTokens",total_tokens)
        messages.append({"role":"system", "content":"Total tokents above max:" + str(total_tokens)})
        break
      

      print("Think " +str(i+1)+ ": ", think["choices"][0]["message"]["tool_calls"][0]["function"]["name"])
      think_args = json.loads(think["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],strict=False)

      thought = think_args["thought"]
      print(thought)
      thoughtAppend = thought + "\nAct "+str(i+1)+ ": "
      messages.append({"role":"user", "content":thoughtAppend})
      
      print("-----------------------------------------------------------------------------------")

      # -------------------------- End think --------------------------
      # -------------------------- Act --------------------------
      act = llm.create_chat_completion(messages=messages, max_tokens=1000, temperature=0.0, top_p=1.0,
                        repeat_penalty=1.1, top_k=1, tools=tools,tool_choice="auto")
      total_tokens = act["usage"]["total_tokens"]
      if(total_tokens > 15000):
        print("totalTokens",total_tokens)
        break
      if act["choices"][0]["finish_reason"] != "stop": #Checks if a tool call has been requested. If not, pass the following and allow the LLM to think again.
        stuck_counter = 0
        print("Act " +str(i+1)+": ", act["choices"][0]["message"]["tool_calls"][0]["function"]["name"])
        act_args = json.loads(act["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],strict=False)
        functionName = act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        # -------------------------- End act --------------------------

        # -------------------------- Observation --------------------------
        if functionName == "get_overview_of_task":
          obs = getOverviewReAct(task)
          actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+thought+"]" + "\nObservation" +str(i+1) + ": "}
          messages.append(actAppend)
          print("-----------------------------------------------------------------------------------")
          print("Observation " +str(i+1)+": ",obs)
          obsAppend = {"role":"user", "content": obs + "\nThought " +str(i+2) +": "} # Skal kun tilføjes hvis der er flere loops rundt.
          messages.append(obsAppend)
        elif functionName == "describe_robots":
          actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"[]" + "\nObservation" +str(i+1) + ": "}
          messages.append(actAppend)
          description = "There are a total of three different robot-types in the environment. They are all mobile robots with different attributes. The first robot-type is a fast_transportation_robot. fast_transportation_robot are very useful if the task is to move between points without manipulating and to transport items on top of the robot. fast_transportation_robot are also faster than the other types of robots. fast_transportation_robot can not manipulate or pick up itmes.\nThe second robot-type is a robot_with_gripper which has a gripper as its tool. robot_with_gripper can only grab solid things. robot_with_gripper can only grab one item at a time.\nThe third robot-type is a robot_with_front_loader which are robots equipped with a front loader shovel. robot_with_front_loader can be used as a shovel, a front loader, a bulldozer, and a heavy lifting mechanism. robot_with_front_loader robots are very useful when a big or heavy item need to be moved. robot_with_front_loader are very good at scooping and picking up piles of items and move the items to desired locations. Only the robots mentioned here are present in the environment."
          obs = "Description: " + description
          print("-----------------------------------------------------------------------------------")
          print("Observation " +str(i+1)+": ",obs)
          obsAppend = {"role":"user", "content": obs + "\nThought " +str(i+2) +": "} # Skal kun tilføjes hvis der er flere loops rundt.
          messages.append(obsAppend)
        elif functionName =="select_robot":
          actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+thought+"]" + "\nObservation" +str(i+1) + ": "}
          messages.append(actAppend)
          obs = act_args
          print("-----------------------------------------------------------------------------------")
          print("Observation " +str(i+1)+": ",obs)
          obsAppend = {"role":"user", "content": str(obs) + "\nThought " +str(i+2) +": "} # Skal kun tilføjes hvis der er flere loops rundt.
          messages.append(obsAppend)
        elif functionName =="extract_coordinates":
          obs = extract_coordinates(llmObject=llm,task=task)
          coordinates = finishCoordinates(obs)
          actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+task+"]" + "\nObservation" +str(i+1) + ": "}
          messages.append(actAppend)
          print("-----------------------------------------------------------------------------------")
          print("Observation " +str(i+1)+": ",obs)
          obsAppend = {"role":"user", "content": str(obs) + "\nThought " +str(i+2) +": "} # Skal kun tilføjes hvis der er flere loops rundt.
          messages.append(obsAppend)
        elif functionName =="finish":
          actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+thought+"]" + "\nObservation" +str(i+1) + ": "}
          messages.append(actAppend)
          obs = act_args
          if coordinates is not None:
            obs["EXTRACTED_COORDINATE(S)"] = coordinates
          obsAppend = {"role":"user", "content": str(obs)} # Skal kun tilføjes hvis der er flere loops rundt.
          messages.append(obsAppend)
          print("-----------------------------------------------------------------------------------")
          print("Final result", obs)
          print("\n")
          break
      else:
        stuck_counter += 1
        print("Stuck Counter", stuck_counter)
        if stuck_counter  == 5:
          print("FORCE FINISH")
          finish = llm.create_chat_completion(messages=messages, max_tokens=1000, temperature=0.0, top_p=1.0,
                        repeat_penalty=1.1, top_k=1, tools=tools,tool_choice={"type": "function", "function": {"name": "finish"}})
          finish_args = json.loads(finish["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],strict=False)
          actAppend = {"role":"user", "content": finish["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+thought+"]" + "\nObservation" +str(i+1) + ": "}
          messages.append(actAppend)
          messages.append({"role":"system", "content": "Force Finish"})
          obs = finish_args
          if coordinates is not None:
            obs["EXTRACTED_COORDINATE(S)"] = coordinates
          obsAppend = {"role":"user", "content": str(obs)} # Skal kun tilføjes hvis der er flere loops rundt.
          messages.append(obsAppend)
          print("-----------------------------------------------------------------------------------")
          print("Final result", obs)
          print("\n")
          break
        elif stuck_counter > 3:
          message = {"role":"user", "content": "Which of the tools could be used in the next ACT phase?"}
          print(message)
          messages.append(message)
      print("-----------------------------------------------------------------------------------")
      i+=1  
      # -------------------------- End observation --------------------------"""

    with open(name, 'a', encoding='UTF8', newline='') as f:
      writer = csv.writer(f)
      for k in range(len(messages)):
        if k == 0:
          writer.writerow([task,"", messages[k]["content"],total_tokens])
        else:
          writer.writerow(["","", messages[k]["content"], ""])

def reActLoop(simObject,llm,task):
  llm.reset()
  messages = [
  {
    "role": "system",
    "content": readPromptsFromTxt("ReAct/reActSystemPrompt.txt")[0]
  },
  {
    "role": "user",
    "content": "Task: " + task + "\nThought 1: " 
  }
  ]
  print("\nCurrent Task:", task)  
  simObject._ui.updateTaskTextbox("Current Task: "+task+"\n")
  total_tokens = 0
  i = 0
  stuck_counter = 0
  coordinates = None
  while True:
    # -------------------------- Think --------------------------
    think = llm.create_chat_completion(messages=messages, max_tokens=1000, temperature=0.0, top_p=1.0,
                      repeat_penalty=1.1, top_k=1, tools=tools,tool_choice={"type": "function", "function": {"name": "think"}})
    
    total_tokens = think["usage"]["total_tokens"]
    if(total_tokens > 15000):
      print("totalTokens",total_tokens)
      messages.append({"role":"system", "content":"Total tokents above max:" + str(total_tokens)})
      break
    
    think_args = json.loads(think["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],strict=False)
    thought = think_args["thought"]
    thoughtAppend = thought + "\nAct "+str(i+1)+ ": "
    messages.append({"role":"user", "content":thoughtAppend})
    print("Thought " +str(i+1)+ ": ",thought)
    simObject._ui.updateTaskTextbox("Thought " +str(i+1)+ ": " +thought +"\n\n")
    
    print("-----------------------------------------------------------------------------------")

    # -------------------------- End think --------------------------
    # -------------------------- Act --------------------------
    act = llm.create_chat_completion(messages=messages, max_tokens=1000, temperature=0.0, top_p=1.0,
                      repeat_penalty=1.1, top_k=1, tools=tools,tool_choice="auto")
    #print(act)
    total_tokens = act["usage"]["total_tokens"]
    if(total_tokens > 15000):
      print("totalTokens",total_tokens)
      break
    if act["choices"][0]["finish_reason"] != "stop": #Checks if a tool call has been requested. If not, pass the following and allow the LLM to think again.
      stuck_counter = 0
      print("Act " +str(i+1)+": ", act["choices"][0]["message"]["tool_calls"][0]["function"]["name"])
      
      simObject._ui.updateTaskTextbox("Act " +str(i+1)+": " + act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"\n")
      
      
      
      act_args = json.loads(act["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],strict=False)
      functionName = act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
      # -------------------------- End act --------------------------

      # -------------------------- Observation --------------------------
      if functionName == "get_overview_of_task":
        obs = getOverviewReAct(task)
        actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+thought+"]" + "\nObservation" +str(i+1) + ": "}
        messages.append(actAppend)
        print("-----------------------------------------------------------------------------------")
        print("Observation " +str(i+1)+": ",obs)
        simObject._ui.updateTaskTextbox("Observation " +str(i+1)+": " + obs+"\n\n")
        obsAppend = {"role":"user", "content": obs + "\nThought " +str(i+2) +": "} # Skal kun tilføjes hvis der er flere loops rundt.
        messages.append(obsAppend)
      elif functionName == "describe_robots":
        actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"[]" + "\nObservation" +str(i+1) + ": "}
        messages.append(actAppend)
        description = "There are a total of three different robot-types in the environment. They are all mobile robots with different attributes. The first robot-type is a fast_transportation_robot. fast_transportation_robot are very useful if the task is to only move between points or to transport items on top of the robot. fast_transportation_robot are also faster than the other types of robots. fast_transportation_robot can not manipulate or pick up itmes.\nThe second robot-type is a robot_with_gripper which has a gripper as its tool. robot_with_gripper can only grab solid things. robot_with_gripper can only grab one item at a time.\nThe third robot-type is a robot_with_front_loader which are robots equipped with a front loader shovel. robot_with_front_loader can be used as a shovel, a front loader, a bulldozer, and a heavy lifting mechanism. robot_with_front_loader robots are very useful when a big or heavy item need to be moved. robot_with_front_loader are very good at scooping and picking up piles of items and move the items to desired locations. Only the robots mentioned here are present in the environment."
        obs = "Description: " + description
        print("-----------------------------------------------------------------------------------")
        print("Observation " +str(i+1)+": ",obs)
        simObject._ui.updateTaskTextbox("Observation " +str(i+1)+": "+obs+"\n\n")
        obsAppend = {"role":"user", "content": obs + "\nThought " +str(i+2) +": "} # Skal kun tilføjes hvis der er flere loops rundt.
        messages.append(obsAppend)
      elif functionName =="select_robot":
        actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+thought+"]" + "\nObservation" +str(i+1) + ": "}
        messages.append(actAppend)
        obs = act_args
        print("-----------------------------------------------------------------------------------")
        print("Observation " +str(i+1)+": ",obs)
        simObject._ui.updateTaskTextbox("Observation " +str(i+1)+": " + str(obs)+"\n\n")
        obsAppend = {"role":"user", "content": str(obs) + "\nThought " +str(i+2) +": "} # Skal kun tilføjes hvis der er flere loops rundt.
        messages.append(obsAppend)
      elif functionName =="extract_coordinates":
        obs = extract_coordinates(llmObject=llm,task=task)
        coordinates = finishCoordinates(obs)
        actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+task+"]" + "\nObservation" +str(i+1) + ": "}
        messages.append(actAppend)
        print("-----------------------------------------------------------------------------------")
        print("Observation " +str(i+1)+": ",obs)
        simObject._ui.updateTaskTextbox("Observation " +str(i+1)+": "+obs+"\n\n")
        obsAppend = {"role":"user", "content": str(obs) + "\nThought " +str(i+2) +": "} 
        messages.append(obsAppend)
        
      elif functionName =="finish":
        actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+thought+"]" + "\nObservation" +str(i+1) + ": "}
        messages.append(actAppend)
        obs = act_args
        if coordinates is not None:
          obs["EXTRACTED_COORDINATE(S)"] = coordinates
        obsAppend = {"role":"user", "content": str(obs)} # Skal kun tilføjes hvis der er flere loops rundt.
        messages.append(obsAppend)
        print("-----------------------------------------------------------------------------------")
        print("Observation " +str(i+1)+": ",obs)
        simObject._ui.updateTaskTextbox("Observation " +str(i+1)+": " +str(obs)+"\n\n")
        print("\n")
        break
    else:
      stuck_counter += 1
      print("Stuck Counter", stuck_counter)
      if stuck_counter  == 5:
        print("FORCE FINISH")
        finish = llm.create_chat_completion(messages=messages, max_tokens=1000, temperature=0.0, top_p=1.0,
                      repeat_penalty=1.1, top_k=1, tools=tools,tool_choice={"type": "function", "function": {"name": "finish"}})
        finish_args = json.loads(finish["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],strict=False)
        actAppend = {"role":"user", "content": finish["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"["+thought+"]" + "\nObservation" +str(i+1) + ": "}
        messages.append(actAppend)
        messages.append({"role":"system", "content": "Force Finish"})
        obs = finish_args
        if coordinates is not None:
          obs["EXTRACTED_COORDINATE(S)"] = coordinates
        print("-----------------------------------------------------------------------------------")
        print("Observation " +str(i+1)+": ",obs)
        simObject._ui.updateTaskTextbox("Observation " +str(i+1)+": " +str(obs)+"\n\n")
        print("\n")
        break
      elif stuck_counter > 3:
        message = {"role":"user", "content": "Which of the tools could be used in the next ACT phase?"}
        print(message)
        messages.append(message)
    print("-----------------------------------------------------------------------------------")
    i+=1  
    # -------------------------- End observation --------------------------
  return obs, messages

def reActThreadFunction(simObject, llm): # OBS FJERN HARDCODED TASK
  while True:
    time.sleep(5)
    newTasks = simObject.getNotStartedTasks()
    if len(newTasks) > 0:
      taskObject = newTasks[0]
      task = taskObject["task"]
      simObject.changeTask(taskID=taskObject["taskID"],robotType=None, waypoints=None,status="Processing", robot=None, IMMessages = [], state = 0)     

      obs, messages = reActLoop(simObject=simObject,task=task,llm=llm)
      simObject.decodeLLMOutput(taskObject,obs)
    if simObject._ui.exit == True:
      break
    
def conversationReAct(simObject,inputString,llmConversation):
  systemPrompt = readPromptsFromTxt("ReAct/conversationSystemPrompt.txt")[0]
  messagesConversation = [
        {
          "role": "user",
          "content": systemPrompt
        },
        {
          "role": "user",
          "content": inputString + "\n Thought: "
        },
  ]
  
  simObject._ui.updateConversationTextbox("User: " + inputString+"\n")
  think = llmConversation.create_chat_completion(messages=messagesConversation, max_tokens=1000, temperature=0.0, top_p=1.0,
                              repeat_penalty=1.1, top_k=1, tools=tools,tool_choice="auto")
            
  think_args = json.loads(think["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],strict=False)
  thought = think_args["thought"]
  print("Thought:",thought)
  simObject._ui.updateConversationTextbox("Thought: " + thought+"\n")
  thoughtAppend = thought + "\nAct : "
  messagesConversation.append({"role":"user", "content":thoughtAppend})


  act = llmConversation.create_chat_completion(messages=messagesConversation, max_tokens=1000, temperature=0.0, top_p=1.0,
                    repeat_penalty=1.1, top_k=1, tools=toolsConversation,tool_choice="auto")
  
  if act["choices"][0]["finish_reason"] != "stop": #Checks if a tool call has been requested. If not, pass the following and allow the LLM to think again.
    print("Action :", act["choices"][0]["message"]["tool_calls"][0]["function"]["name"])
    simObject._ui.updateConversationTextbox("Action: "+ act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"\n")
    functionName = act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]

    if functionName == "get_status_of_tasks":
      actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"[]"}
      messagesConversation.append(actAppend)
      obs = simObject.getTasks()
      print(obs)
      if len(obs) > 0:
        for task in obs:
          currentTask = "Task " +str(int(task["taskID"])+1)+": " + task["task"] + ", Status: " + task["status"]+"\n"
          simObject._ui.updateConversationTextbox(currentTask)
      else:
        simObject._ui.updateConversationTextbox("No tasks are available. Start a task and try again.\n")

    elif functionName == "get_status_of_robots":
      actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"[]"}
      messagesConversation.append(actAppend)
      obs = simObject.getStatesOfAllRobots()
      print(obs)
      number = 1
      for robotType in obs:
        name = robotType[0]
        for robot in robotType[1]:
          simObject._ui.updateConversationTextbox("Robot "+str(number)+", Type: " + name + ", Status: " +str(robot)+"\n")
          number+=1
      
    elif  functionName == "get_position_of_robots":
      actAppend = {"role":"user", "content": act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]+"[]"}
      messagesConversation.append(actAppend)
      obs = simObject.getPositionsOfAllRobots()
      print(obs)
      number = 1
      for robotType in obs:
        name = robotType[0]
        for robot in robotType[1]:
          simObject._ui.updateConversationTextbox("Robot "+str(number)+", Type: " + name + ", Position: " +str(robot)+"\n")
          number+=1
  simObject._ui.updateConversationTextbox("\n")
  return obs,messagesConversation
    
def conversationThreadFunction(simObject,llm):
  while True:
    time.sleep(5)
    if len(simObject.getConversationList()) > 0:
      currentConversation = simObject.getConversationList().pop(0)
      obs, messages = conversationReAct(simObject=simObject,inputString=currentConversation,llmConversation=llm)
    if simObject._ui.exit == True:
      break

def IMExample():
  llm = Llama(
    model_path=modelPath,
    chat_format="chatml-function-calling",
    seed = 1,
    n_threads=4, # CPU cores
    n_ctx = 1024, # Allows for larger message history
    n_batch= 512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=-1, # Change this value based on your model and your GPU VRAM pool.
    verbose=False # True for messages from Llama)
  )
  llm.reset()
  task = "Pick up leaves at (3, 3), pass through (4, 4) and place the leaves at (5, 5)."
  pos = (1,1)
  sceneDetector= ["nothing", "leaves"]

  functionName = "None"
  state = 0
  succes = ""
  itemName = ""
  messages = [
  {
    "role": "system",
    "content": "An overview and steps of a task is given to you. Your job is only to choose the correct function, depeding on the step in the overview."
  },
  {
    "role": "user",
    "content": "Task: " + task 
  }
  ]
  print(messages[-1]["content"], "\n")
  
  obs = overviewIM(task=task, robotPos=pos)
  messages.append({
          "role": "user",
          "content": "Overview with steps of the task: \"" + obs + "  \""})
  
  messages.append({
          "role": "user",
          "content": "Please choose the correct function for step " + str(state+1)+ " in the overview."})
  print(messages[-1]["content"])
  
  while True:

    act = llm.create_chat_completion(messages=messages, max_tokens=1000, temperature=0.0, top_p=1.0,
                      repeat_penalty=1.1, top_k=1, tools=toolsIM,tool_choice="auto")
    
    messages.pop()
    
    functionName = act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]

    print(functionName)
    
    if functionName == "move_robot":
      messages.append({
          "role": "user",
          "content": "Please choose the correct function for step " + str(state+2) + " in the overview."})
      print(messages[-1]["content"])
    
    elif functionName == "pick_up_object":
        messagesPickUp = [
        {
          "role": "user",
          "content": "A task and a list of objects are given to you. You should choose the most appropriate object for the task. If there are no appropriate object, you should just choose \'nothing\' from the list of objects. An example could be, Task: Move banana from 1,3 to 2,7. List of objects: [\'nothing\', \'banana\']. The chosen object should be banana, since the task says so. Another example could be, Task: Move banana from 2,6 to 1,2. List of objects: [\'nothing\', \'apple\']. The chosen object should be nothing, since there are no appropriate object in the list of objects."
        },
        {
          "role": "user",
          "content": "Task: " + task + "List of objects: " + str(sceneDetector)
        }
        ]
        act = llm.create_chat_completion(messages=messagesPickUp, max_tokens=1000, temperature=0.0, top_p=1.0,
                      repeat_penalty=1.1, top_k=1, tools=toolsIM,tool_choice={"type": "function", "function": {"name": "pick_up_object"}})
        succes = succesDetector(0.8)
        
        if succes == True:
          act_args = json.loads(act["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],strict=False)
          itemName = act_args["object"]
          print("Action: Succes")
          print("Robot picked up:",itemName)
          messages.append({
            "role": "user",
            "content": "Please choose the correct function for step " + str(state+2)+ " in the overview."})
          print(messages[-1]["content"])
        else:
          print("Action: Failed")
          state -= 1
          messages.append({
            "role": "user",
            "content": "Please choose the correct function for step " + str(state+2) + " in the overview."})
          print(messages[-1]["content"])
      
    elif functionName == "place_object":
        succes = succesDetector(0.9)
        if succes == True:
          print("Action: Succes")
          print("Robot placed:",itemName)
          messages.append({
            "role": "user",
            "content": "Please choose the correct function for step " + str(state+2)+ " in the overview."})
          print(messages[-1]["content"])
        else:
          print("Action: Failed")
          state -= 1
          messages.append({
            "role": "user",
            "content": "Please choose the correct function for step " + str(state+2)+ " in the overview."})
          print(messages[-1]["content"])
    if functionName == "task_completed":
        break
    state += 1
       
def fullSystem():  
  ReActllm = Llama(
    model_path=modelPath,
    chat_format="chatml-function-calling",
    seed = 1,
    n_threads=10, # CPU cores
    n_ctx = 4*4096, # Allows for larger message history
    n_batch= 4*2048, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=-1, # Change this value based on your model and your GPU VRAM pool.
    verbose=False # True for messages from Llama)
  )  
  
  Conversationllm = Llama(
    model_path=modelPath,
    chat_format="chatml-function-calling",
    seed = 1,
    n_threads=2, # CPU cores
    n_ctx = 1024, # Allows for larger message history
    n_batch= 512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=-1, # Change this value based on your model and your GPU VRAM pool.
    verbose=False # True for messages from Llama)
  )  
  
  
  IMllm = Llama(
    model_path=modelPath,
    chat_format="chatml-function-calling",
    seed = 1,
    n_threads=4, # CPU cores
    n_ctx = 1024, # Allows for larger message history
    n_batch= 512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=-1, # Change this value based on your model and your GPU VRAM pool.
    verbose=False # True for messages from Llama)
  )  
  
  runSim = True
  obstacles = [(0,4),(1,4),(2,4),(3,4)]
  sim = simulation.Simulation(createSim=runSim, mapSize=(7,7),randomPlacement=False,wantWaitzones=False,obstacles=obstacles,amountFast=2,amountGripper=2,amountFrontLoader=2, IMllm = IMllm)
  reActThread = threading.Thread(target=reActThreadFunction, args=(sim,ReActllm))
  simThread = threading.Thread(target=sim.sim)
  conversationThread = threading.Thread(target=conversationThreadFunction,args = (sim,Conversationllm))

  simThread.start()
  reActThread.start()
  conversationThread.start()  
    
if __name__ == '__main__':

  #modelCapabilitiesTest()
  #standardApproachTest()
  #chainOfThoughtApproachTest()
  #reActApproachTest()
  #IMExample()
  fullSystem()
