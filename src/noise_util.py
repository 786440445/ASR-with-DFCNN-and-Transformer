import shutil, os
import random
from tqdm import tqdm
os.sys.path.append('../')
from src.noise import add_noise
from src.train import prepare_data
from src.const import Const


def delete_files(pathDir):
    fileList = list(os.listdir(pathDir))
    for file in fileList:
        file = os.path.join(pathDir, file)
        if os.path.isfile(file):
            os.remove(file)
        else:
            shutil.rmtree(file)
    print("delete noise data successfully")


def main():
    rate = 1
    out_path = Const.NoiseOutPath
    delete_files(out_path)

    train_data = prepare_data('train', batch_size=5, is_shuffle=False, length=None)
    pathlist = train_data.path_lst
    pylist = train_data.pny_lst
    hzlist = train_data.han_lst
    length = len(pathlist)
    rand_list = random.sample(range(length), int(rate * length))

    pre_list = []
    for i in rand_list:
        path = pathlist[i]
        pre_list.append(os.path.join(Const.SpeechDataPath, path))
    _, filename_list = add_noise(pre_list, out_path=Const.NoiseOutPath, keep_bits=False)

    data = ''
    with open(Const.NoiseDataTxT, 'w') as f:
        for i in range(len(rand_list)):
            pinyin = pylist[rand_list[i]]
            hanzi = hzlist[rand_list[i]]
            data += filename_list[i] + '\t' + pinyin + '\t' + hanzi + '\n'
        f.writelines(data[:-1])
    print('---------------噪声数据生成完毕------------')


if __name__ == '__main__':
    main()