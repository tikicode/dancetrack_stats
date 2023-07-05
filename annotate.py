import cv2 
import os
import sys

"""
1. Allow user to select which 
sequence to label. Ask for 
frame or to start from scratch. 
2. Load in frames and data
3. Display data. Grab number of 
dancers from gt.txt 
4. Generate ordering by asking 
user to compare two dancers one 
by one. Map keys for closer, farther
same distance, or unsure. 
5. Keys map for going to previous 
or next frame but check if gt exists 
and allow skipping only if exists
"""