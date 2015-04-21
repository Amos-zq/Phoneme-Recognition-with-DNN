# coding=UTF-8

#python predict.py [test_data] [input_model_name]

from DNN import predict
import sys
import time
import os

f = open("./transfer table/48_toNum.map", "r")
ff = open("./transfer table/48_39.map", "r")
of = open(str(int(time.time())), "w")

#num轉48
num = {}
for line in f:
	arr = line.rstrip("\n").split("\t")
	num[arr[1]] = arr[0]

#48轉39
big = {}
for line in ff:
	arr = line.rstrip("\n").split("\t")
	big[arr[0]] = arr[1]

#output成可上傳kaggle的格式
of.write("Id,Prediction\n")
#predict(  test_data, input_model_name) ; ans = [ ("pid", "label") ]
ans = predict(sys.argv[1], sys.argv[2])
for a in ans:
	of.write(a[0]+","+big[num[a[1]]]+"\n")

#close files
f.close()
ff.close()
of.close()
