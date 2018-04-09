# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 07:44:31 2018

@author: Milind
"""
 

def type_1 ():
 

  for i in range (1,100):
      print(i)
      
      for m in range(0,3):
         x=i
         y = i*10
         yield (x, y)
        
oksir = type_1()

fp =10
tn = 90
        
        