f = open('./train.txt', 'r', encoding='utf-8')
while True:
	content = f.readline()
	if content == '':
		break
	content = content.strip().split()
	# get entity name
	en1_idx = content[0]
	en2_idx = content[1]
	list_word = list(content[3:])
	en1 = list_word[int(en1_idx)]
	en2 = list_word[int(en2_idx)]
	sentence = " ".join(content[3:])
	print(en1)
	print(en2)
	print(sentence)