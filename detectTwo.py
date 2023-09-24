# -*- coding: utf-8 -*-
import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
import YB_Pcb_Car
import time

car = YB_Pcb_Car.YB_Pcb_Car()
car.Ctrl_Servo(1, 90)
car.Ctrl_Servo(2, 130)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

ACTIONS = 4

parking_lot_width = 10
parking_lot_length = 10

EPISODES = 15
MAX_STEPS = 10

LEARNING_RATE = 0.81
GAMMA = 0.96

Parked_State_Reward = 188200
#----------------------------------tensor_code---------------------
def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)
  count = int(get_output_tensor(interpreter, 2))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

#---------------------end_tensor-----------------

def step(state, action):
  if(action == 0):
    state += parking_lot_width
  elif(action == 1):
    state -= parking_lot_width
  elif(action == 2):
    state -= 1
  elif(action == 3):
    state += 1
  return state

def reset():
  while(True):
    keyValue = cv2.waitKey(10)
    if keyValue == ord('q'):
      break#(5, 0)
  return 5



def act_validation(action, state):
  global parking_lot_width, parking_lot_length
  if(state // parking_lot_width == 0 and action == 1):
    return False
  elif(state // parking_lot_width == parking_lot_length - 1 and action == 0):
    return False
  elif(state % parking_lot_length == 0 and action == 2):
    return False
  elif(state % parking_lot_length == parking_lot_width - 1 and action == 3):
    return False
  else:
    return True

def main():
  global parking_lot_width, parking_lot_length
  STATES = parking_lot_width * parking_lot_length
  Q = np.zeros((STATES, ACTIONS))
  epsilon = 0.9
  labels = load_labels()
  interpreter = Interpreter('detect.tflite')
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  cap = cv2.VideoCapture(0)
  while cap.isOpened():
    for episode in range(EPISODES):
      state = 5
      for _ in range(MAX_STEPS):

        if np.random.uniform(0, 1) < epsilon:
          action_valid = False
          while(not action_valid):
             action =  int(3 * np.random.uniform(0, 1))
             action_valid = act_validation(action, state)

        else:
          action = np.argmax(Q[state, :])

        car.Car_Action(action)
        next_state = step(state, action)

        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        res = detect_objects(interpreter, img, 0.8)
        print(res)
        resMat = np.zeros((len(res), 4))

        for result in res:
          x = int(0)
          ymin, xmin, ymax, xmax = result['bounding_box']
          resMat[x][0] = int(max(1,xmin * CAMERA_WIDTH))
          resMat[x][1] = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
          resMat[x][2] = int(max(1, ymin * CAMERA_HEIGHT))
          resMat[x][3] = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
          x += 1
          cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
          cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)

        cv2.imshow('Pi Feed', frame)

        small = 0
        if(len(res) == 0):
          reward = 0
          done = False
        else:
          for i in (0, len(res) - 1):
            if((resMat[small][0] - resMat[small][1]) **2 + (resMat[small][2] - resMat[small][3]) ** 2 > (resMat[i][0] - resMat[i][1]) **2 + (resMat[i][2] - resMat[i][3]) ** 2):
              small = i

          reward = ((resMat[small][0] - resMat[small][1]) **2 + (resMat[small][2] - resMat[small][3]) ** 2) / Parked_State_Reward#직접 해본다음 안다.
          done = (reward > 0.99)

        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state
        time.sleep(8)
        if cv2.waitKey(10) & 0xFF ==ord('q'):
          cap.release()
          cv2.destroyAllWindows()
      if done:
        #rewards.append(reward)#unnecessary
        epsilon -= 0.001
        break  # reached goal


if __name__ == "__main__":
    main()
