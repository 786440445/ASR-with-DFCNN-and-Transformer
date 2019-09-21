import argparse


class AmHparams:
    # 声学模型参数
    parser = argparse.ArgumentParser()
    # 初始学习率为0.001,10epochs后设置为0.0001
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--gpu_nums', default=1, type=int)
    parser.add_argument('--is_training', default=True, type=bool)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--feature_dim', default=200, type=int)
    # 声学模型最大帧数
    parser.add_argument('--feature_max_length', default=1600, type=int)


class LmHparams:
    # 语言模型参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', default=200, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_blocks', default=6, type=int)
    parser.add_argument('--position_max_length', default=100, type=int)
    parser.add_argument('--hidden_units', default=512, type=int)
    parser.add_argument('--lr', default=0.0003, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--is_training', default=True, type=bool)
    parser.add_argument('--count', default=500, type=int)


class DataHparams:
    parser = argparse.ArgumentParser()
    parser.add_argument('--thchs30', default=True, type=bool)
    parser.add_argument('--aishell', default=True, type=bool)
    parser.add_argument('--prime', default=True, type=bool)
    parser.add_argument('--stcmd', default=True, type=bool)
    parser.add_argument('--aidatatang', default=False, type=bool)

    parser.add_argument('--noise', default=False, type=bool)
    parser.add_argument('--pinyin_dict', default='../mixdict.txt', type=str)
    parser.add_argument('--hanzi_dict', default='../hanzi.txt', type=str)

    # Low Frame Rate (stacking and skipping frames)
    parser.add_argument('--lfr_m', default=4, type=int, help='Low Frame Rate: number of frames to stack')
    parser.add_argument('--lfr_n', default=3, type=int, help='Low Frame Rate: number of frames to skip')


# 直接使用transformer构建语音识别参数
class TransformerHparams:
    # 声学模型参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_blocks', default=6, type=int)
    parser.add_argument('--position_max_length', default=1000, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--lr', default=0.0003, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--is_training', default=True, type=bool)
    parser.add_argument('--feature_dim', default=80, type=int)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    # 预测长度
    parser.add_argument('--count', default=500, type=int)
    parser.add_argument('--concat', default=4, type=int)
