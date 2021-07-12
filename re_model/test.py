from pprint import pprint

import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from sklearn.metrics import average_precision_score
import argparse

FLAGS = tf.app.flags.FLAGS


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


def main_for_evaluation():
    pathname = "./re_model/model/ATT_GRU_model-"

    wordembedding = np.load('./re_model/data/vec.npy')

    test_settings = network.Settings()
    test_settings.vocab_size = 14123
    test_settings.num_classes = 2
    test_settings.big_num = 5561

    big_num_test = test_settings.big_num

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

           
            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)

        
            #testlist = range(1000, 1800, 100)
            testlist = [450]
            
            for model_iter in testlist:
                # for compatibility purposes only, name key changes from tf 0.x to 1.x, compat_layer
                saver.restore(sess, pathname + str(model_iter))


                time_str = datetime.datetime.now().isoformat()
                print(time_str)
                print('Evaluating all test data and save data for PR curve')

                test_y = np.load('./re_model/data/testall_y.npy')
                test_word = np.load('./re_model/data/testall_word.npy')
                test_pos1 = np.load('./re_model/data/testall_pos1.npy')
                test_pos2 = np.load('./re_model/data/testall_pos2.npy')
                allprob = []
                acc = []
                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    prob, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                    prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
                    for single_prob in prob:
                        allprob.append(single_prob[1:])
                allprob = np.reshape(np.array(allprob), (-1))
                order = np.argsort(-allprob)

                print('saving all test result...')
                current_step = model_iter

                
                np.save('./re_model/out/allprob_iter_' + str(current_step) + '.npy', allprob)
                allans = np.load('./re_model/data/allans.npy')

                # caculate the pr curve area
                average_precision = average_precision_score(allans, allprob)
                print('PR curve area:' + str(average_precision))


def main(_):

    #If you retrain the model, please remember to change the path to your own model below:
    pathname = "./re_model/model/RE_model-450"
    
    wordembedding = np.load('./re_model/data/vec.npy')
    test_settings = network.Settings()
    test_settings.vocab_size = 14123
    test_settings.num_classes = 2
    test_settings.big_num = 1
    
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy
            
            
            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)
            saver.restore(sess, pathname)
            
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
            
            print('reading relation to id')
            relation2id = {'N': 0, 'Y': 1}
            id2relation = {0: 'N', 1: 'Y'}
            
            if not(args.input == None or args.output == None):
                out = open(args.output, "w")
                re_sentences = open(args.input, "r").read()
                sentence_tags = re_sentences.split("---------")
                for sentence_tag in sentence_tags:
                    if sentence_tag:
                        sentence_t = sentence_tag.split("\n")
                        if (sentence_t[0]) == "No relationship detected!":
                            out.write(re_sentences)
                            break
                        else:
                            sentence = sentence_t[0]
                            out.write(sentence+"\n")
                            tag_sens = sentence_t[1:]
                            for tag_sen in tag_sens:
                                en1_en2_sen = tag_sen.strip().split()
                                label = en1_en2_sen[0]
                                en1 = en1_en2_sen[1]
                                en2 = en1_en2_sen[2]
                                sentence = list(en1_en2_sen[3:])
                                # print("name1: " + en1)
                                # print("name2: " + en2)
                                # print(" ".join(sentence))
                                relation = 0
                                en1pos = sentence.index(en1)
                                en2pos = sentence.index(en2)
                                output = []
                                # length of sentence is 70
                                fixlen = 72
                                # max length of position embedding is 60 (-60~+60)
                                maxlen = 60

                                #Encoding test x
                                en1_en2_appear_1 = 0
                                for i in range(fixlen):
                                    if i in [en1pos, en2pos]:
                                        en1_en2_appear_1 += 1
                                        continue
                                    word = word2id['BLANK']
                                    rel_e1 = pos_embed(i - en1pos)
                                    rel_e2 = pos_embed(i - en2pos)
                                    output.append([word, rel_e1, rel_e2])
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
                                test_x = []
                                test_x.append([output])
                                
                                #Encoding test y
                                test_y=[[0,1]]


                                test_x = np.array(test_x)
                                test_y = np.array(test_y)
                                test_word = []
                                test_pos1 = []
                                test_pos2 = []

                                for i in range(len(test_x)):
                                    word = []
                                    pos1 = []
                                    pos2 = []
                                    for j in test_x[i]:
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
                                
                                #print("test_word Matrix:")
                                #print(test_word)
                                #print("test_pos1 Matrix:")
                                #print(test_pos1)
                                #print("test_pos2 Matrix:")
                                #print(test_pos2)
                                
                                pred, accuracy = test_step(test_word, test_pos1, test_pos2, test_y)
                                pred = np.reshape(np.array(pred), (1, test_settings.num_classes))[0]
                                print("relationship: ")
                                #print(prob)
                                pre = pred.argsort()[-2:][::-1]
                                print(id2relation[pre[0]]+"(" + en1 + "," + en2 + ")"+"\n")
                                entity1 = " ".join(en1.split("_"))
                                entity2 = " ".join(en2.split("_"))
                                if id2relation[pre[0]]=="N":
                                    out.write("No relationship" +"(" + entity1 + "," + entity2 + ")"+"\n")
                                else:
                                    if label == "ORG":
                                        out.write("Has product" +"(" + entity1 + "," + entity2 + ")"+"\n")
                                    elif label == "VER":
                                        out.write("Is version of" +"(" + entity1 + "," + entity2 + ")"+"\n")
                                    else:
                                        out.write("Has vulnerability" +"(" + entity1 + "," + entity2 + ")"+"\n")
                            out.write("---------\n")  
                            
                            
                            #result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                            #print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a RE model')
    parser.add_argument('-input', metavar='Input file', default=None,
                        help='(dir for the file input)')
    parser.add_argument('-output', metavar='Ouput file', default=None,
                        help='(Output file)')
    args = parser.parse_args()
    main(args)