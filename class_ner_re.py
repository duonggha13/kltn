from utils import clean
import os
from pre_re import preprocess_re

file_path_ner_output = ""

def ner(file_path_input):
	global file_path_ner_output
	file_out_class = "./file/out_class.txt"
	file_in = open(file_path_input, "r")
	sentences = file_in.read().splitlines()
	file_in.close()
	file_in_class = "./file/input.txt"
	in_class = open(file_in_class, "w")
	for sentence in sentences:
		sentence = clean(sentence)
		in_class.write(sentence + "\n")
	in_class.close()
	os.system('python class_eval.py -model checkpoints/1_5_2105015557/ -input '+ file_in_class+' -output '+file_out_class)
	f = open(file_out_class, "r")
	fin_ner = "./file/file_input_ner.txt"
	in_ner = open(fin_ner, "w")
	lines = f.read().splitlines()
	for line in lines:
		a = line.split("\t")
		if a[1] == "1":
			in_ner.write(a[0] + "\n")
		else:
			in_ner.write("")
	in_ner.close()
	fout_ner = "./file/out_ner.txt"
	os.system('python ner_eval.py -model checkpoints/1_5_NER_210501235625/ -input '+ fin_ner+' -output '+fout_ner)
	f = open(fout_ner,"r")
	file_path_ner_output = fout_ner
	output_ner = f.read()
	return output_ner

def re_from_file(file_path_input):
	ner(file_path_input)
	file_path_input_re = preprocess_re(file_path_ner_output)
	file_out_re = "./file/out_re.txt"
	os.system('python re_model/test.py -input '+ file_path_input_re+' -output '+file_out_re)
	output_re = open(file_out_re, "r").read()
	return output_re