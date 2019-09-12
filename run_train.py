#-*- coding: UTF-8 -*-

import os
import sys
import logging
import master


def main():

    m = master.Master('conf.json')
    vocab = m.load_vocab()
    m.build_emb(vocab)
    m.load_data()
    print('test_hi')
    m.creat_graph()
    print('test_hi_2')
    m.train()
    print('DONE TRAINING')
    logging.info("Done Train !")


if __name__ == '__main__':
    if len(sys.argv) != 1:
        exit(1)
    else:
        # logging
        fileHandler = logging.FileHandler(os.path.abspath('.')+'/log.train', mode='w', encoding='UTF-8')
        formatter = logging.Formatter('%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
        fileHandler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        
        main()


