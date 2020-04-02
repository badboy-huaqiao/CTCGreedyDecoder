# Copyright huaqiaoz

# DEFINE_string(m_td, "", text_detection_model_message);
# DEFINE_string(m_tr, "", text_recognition_model_message);
# DEFINE_string(m_tr_ss, "0123456789abcdefghijklmnopqrstuvwxyz", text_recognition_model_symbols_set_message);

import  numpy as np
import math,sys

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
pay_symbol = "#"

kPadSymbol = '#'

kAlphabet = alphabet + kPadSymbol

conf = 1.0

argmax = None
prob = None

cache = None

def softmax(begin, end, argmax, prob):
    max = None
    if begin >= end:
        max = begin
    else:
        max = end
    distance = None 
    if cache.index(max) >= cache.index(begin):
        distance = cache.index(max) - cache.index(begin)
    else:
        distance = cache.index(max) - cache.index(begin)
    
    max_val = max
    
    sum = 0
    i = begin
    while i != end:
        sum += np.exp(i-max_val)
        i+=1
    if math.fabs(sum) < sys.float_info.min:
        print("sum can't be equal to zero") 
    
    prob = 1.0 / float(sum)
    
        
    
def CTCGreedyDecoder(data, alphabet, pad_symbol, conf):
    cache = data
    res = ""
    prev_pad = False
    conf = 1
    
    num_classes = len(alphabet)
    for index in range(len(data)):
        softmax(data[index],index+num_classes,argmax,prob)
        conf *= prob
        
        symbol = alphabet[argmax]
        if symbol != pad_symbol:
            if res == "" or prev_pad or (res != "" and symbol != res[-1]):
                prev_pad = False
                res += symbol
            else:
                prev_pad = True
    return res