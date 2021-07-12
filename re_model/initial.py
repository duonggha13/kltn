import numpy as np
import os


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


# reading data
def init():
    print('reading word embedding data...')
    vec = []
    word2id = {}
    f = open('./re_model/origin_data/vec.txt', encoding='utf-8')
    content = f.readline()
    content = content.strip().split()
    dim = int(content[1])
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [(float)(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    for i in range(13545,15000):
        vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
        vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

    print('reading relation to id')

    # length of sentence is 70
    fixlen = 72
    # max length of position embedding is 60 (-60~+60)
    maxlen = 60
    relation2id = {'N': 0, 'Y': 1}
    id2relation = {0: 'N', 1: 'Y'}
    train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector

    print('reading train data...')
    f = open('./re_model/origin_data/train.txt', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split()
        # get entity name
        en1_idx = int(content[0])
        en2_idx = int(content[1])
        list_word = list(content[3:])
        en1 = list_word[int(en1_idx)]
        #en1 = " ".join(en1.split("_"))
        en2 = list_word[int(en2_idx)]
        #en2 = " ".join(en2.split("_"))
        relation = 0
        relation = relation2id[content[2]]
        # put the same entity pair sentences into a dict
        tup = (en1, en2)
        label_tag = 0
        if tup not in train_sen:
            train_sen[tup] = []
            train_sen[tup].append([])
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            train_ans[tup] = []
            train_ans[tup].append(label)
        else:
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1

            temp = find_index(label, train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                label_tag = len(train_ans[tup]) - 1
                train_sen[tup].append([])
            else:
                label_tag = temp

        sentence = content[3:]

        en1pos = en1_idx
        en2pos = en2_idx
        
        output = []

        #Embeding the position
        en1_en2_appear_1 = 0
        for i in range(fixlen):
            if i in [en1pos, en2pos]:
                en1_en2_appear_1 += 1
                continue
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])
        if en1_en2_appear_1 != 2:
            logging.info('error append pos ' + str(en1_en2_appear_1) + ' ' + str(en1pos) + ' ' + str(en2pos) + ' ' + str(relation))

        en1_en2_appear_2 = 0
        for i in range(min(fixlen, len(sentence))):
            if i in [en1pos, en2pos]:
                en1_en2_appear_2 += 1
                continue
            word = 0
            if sentence[i] not in word2id:
                ps = sentence[i].split('_')
                # avg_vec = np.zeros(commons.word_emb_dim)
                c = 0
                for p in ps:
                    if p in word2id:
                        c += 1
                        # avg_vec += vec[word2id[p]]
                if c > 0:
                    # avg_vec = avg_vec / c
                    word2id[sentence[i]] = len(word2id)
                    #vec.append(avg_vec)
                else:
                    word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            output[i-en1_en2_appear_2][0] = word

        if en1_en2_appear_2 != 2:
            logging.info('error append word ' + str(en1_en2_appear_2) + ' ' + str(en1pos) + ' ' + str(en2pos) + ' ' + str(relation) + ' ' + str(list(range(min(fixlen, len(sentence))))) + ' || ' + ' '.join(content))

        train_sen[tup][label_tag].append(output)

    print('reading test data ...')

    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)

    f = open('./re_model/origin_data/test.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split()

        en1pos = int(content[0])
        en2pos = int(content[1])

        relation = relation2id[content[2]]

        sentence = content[3:]

        en1 = ''
        en2 = ''

        for i in range(len(sentence)):
            if i == en1pos:
                en1 = sentence[i]
            if i == en2pos:
                en2 = sentence[i]

        tup = (en1, en2)

        if tup not in test_sen:
            test_sen[tup] = []
            y_id = relation
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            test_ans[tup] = label
        else:
            y_id = relation
            test_ans[tup][y_id] = 1

        output = []

        en1_en2_appear = 0
        for i in range(fixlen):
            # logging.info(i)
            if i in [en1pos, en2pos]:
                en1_en2_appear += 1
                continue
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])
        if en1_en2_appear != 2:
            logging.info('error append pos test ' + str(en1_en2_appear) + ' ' + str(en1pos) + ' ' + str(en2pos) + ' ' + str(relation))

        en1_en2_appear = 0
        for i in range(min(fixlen, len(sentence))):
            if i in [en1pos, en2pos]:
                en1_en2_appear += 1
                continue
            word = 0
            if sentence[i] not in word2id:
                ps = sentence[i].split('_')
                # avg_vec = np.zeros(commons.word_emb_dim)
                c = 0
                for p in ps:
                    if p in word2id:
                        c += 1
                        # avg_vec += vec[word2id[p]]
                if c > 0:
                    # avg_vec = avg_vec / c
                    word2id[sentence[i]] = len(word2id)
                    # vec.append(avg_vec)
                else:
                    word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i - en1_en2_appear][0] = word
        if en1_en2_appear != 2:
            logging.info('error append word test ' + str(en1_en2_appear) + ' ' + str(en1pos) + ' ' + str(en2pos) + ' ' + str(relation))

        test_sen[tup].append(output)
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    if not os.path.exists("data"):
        os.makedirs("data")

    print('organizing train data')
    f = open('./re_model/data/train_q&a.txt', 'w', encoding='utf-8')
    temp = 0
    for i in train_sen:
        if len(train_ans[i]) != len(train_sen[i]):
            print('ERROR')
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    f.close()

    print('organizing test data')
    f = open('./re_model/data/test_q&a.txt', 'w', encoding='utf-8')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    f.close()

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    np.save('./re_model/data/vec.npy', vec)
    np.save('./re_model/data/train_x.npy', train_x)
    np.save('./re_model/data/train_y.npy', train_y)
    np.save('./re_model/data/testall_x.npy', test_x)
    np.save('./re_model/data/testall_y.npy', test_y)

   


def seperate():
    print('reading training data')
    x_train = np.load('./re_model/data/train_x.npy', allow_pickle=True)

    train_word = []
    train_pos1 = []
    train_pos2 = []

    print('seprating train data')
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_train[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save('./re_model/data/train_word.npy', train_word)
    np.save('./re_model/data/train_pos1.npy', train_pos1)
    np.save('./re_model/data/train_pos2.npy', train_pos2)

    print('seperating test all data')
    x_test = np.load('./re_model/data/testall_x.npy', allow_pickle=True)
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./re_model/data/testall_word.npy', test_word)
    np.save('./re_model/data/testall_pos1.npy', test_pos1)
    np.save('./re_model/data/testall_pos2.npy', test_pos2)



# get answer metric for PR curve evaluation
def getans():
    test_y = np.load('./re_model/data/testall_y.npy', allow_pickle=True)
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    np.save('./re_model/data/allans.npy', allans)


def get_metadata():
    fwrite = open('./re_model/data/metadata.tsv', 'w', encoding='utf-8')
    f = open('./re_model/origin_data/vec.txt', encoding='utf-8')
    f.readline()
    while True:
        content = f.readline().strip()
        if content == '':
            break
        name = content.split()[0]
        fwrite.write(name + '\n')
    f.close()
    fwrite.close()


init()
seperate()
getans()
get_metadata()
