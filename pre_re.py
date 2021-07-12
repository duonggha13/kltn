def preprocess_re(file_ner_out):
	file = file_ner_out
	fin = "./file/in_re.txt"
	file_in_re = open(fin,"w")
	ner = open(file, "r").read()
	sentence_tags = ner.split("---------")
	re = 0
	for sentence_tag in sentence_tags:
		tag_words = []
		org = []
		pro = []
		ver = []
		vul = []
		a = ""
		if len(sentence_tag)>0:
			sentence_t = sentence_tag.split("\n")
			if len(sentence_t[0]) > 0:
				sentence = sentence_t[0]
				tag_words = sentence_t[1:]
			else:
				sentence = sentence_t[1]
				tag_words = sentence_t[2:]
			sentence_origin = sentence
			for tag_word in tag_words:
				sen_merg = sentence
				tag = tag_word.split(" ")[0]
				word = tag_word.split(" ")[1:]
				token = " ".join(word)
				#sentence = sen_merg.replace(" ".join(word), token)
				if tag == "ORG:":
					org.append(token)
				if tag == "PRO:":
					pro.append(token)
				if tag == "VER:":
					ver.append(token)
				if tag == "VUL:":
					vul.append(token)
			org = list(set(org))
			pro = list(set(pro))
			ver = list(set(ver))
			vul = list(set(vul))
			# for i in range(0, len(ver)-1):
			# 	vn = ver[i] + " " + ver[i+1]
			# 	if vn in sentence and ver[i] == "before":
			# 		ver[i:i+2] = [' '.join(ver[i:i+2])]
			# print(ver)
			for m in org:
				for n in pro:
					m1 = "_".join(m.split(" "))
					n1 = "_".join(n.split(" "))
					sen = sentence.replace(m, m1)
					sen1 = sen.replace(n, n1)
					a = a + ("ORG " + m1+ " " + n1+ " " + sen1 + "\n")
			for m in ver:
				for n in pro:
					m1 = "_".join(m.split(" "))
					n1 = "_".join(n.split(" "))
					sen = sentence.replace(m, m1)
					sen1 = sen.replace(n, n1)
					a = a + ("VER " + m1 + " " + n1+ " " + sen1 + "\n")
			for m in pro:
				for n in vul:
					m1 = "_".join(m.split(" "))
					n1 = "_".join(n.split(" "))
					sen = sentence.replace(m, m1)
					sen1 = sen.replace(n, n1)
					a = a + ("VUL " + m1+ " " + n1+ " " + sen1 + "\n")
		if a:
			re = re + 1
			file_in_re.write(sentence_origin + "\n")
			file_in_re.write(a + "---------\n")
	if re==0:
		file_in_re.write("No relationship detected!"+"\n")
		file_in_re.write(ner)
	return fin