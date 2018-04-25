import numpy as np
from matplotlib import pyplot
import os
from multiprocessing import Process, Pool
from scipy import interpolate
import traceback
import time


def lagrange(img, x, y, k, y0):
    t = 0.0
    for i in range(4):  # 三次拉格朗日插值
        u = 1.0
        for j in range(4):
            if i != j:
                u *= (y - y0 - j) / (i - j)
        t += u * img[x, y0 + i, k]
    return t


def get_var_list(img, color):
    m, n, c = img.shape
    var_array = np.zeros((m, n))
    for j in range(n):
        if j + 4 > n:
            var_array[::, j] = var_array[::, j - 1]
            continue
        for i in range(m):
            dot_list = []
            for l in range(4):
                dot_list.append(img[i, j + l, color])
            var_array[i, j] = np.var(dot_list)
    return var_array


def single_process(img, l, k):
    print('process id =', os.getpid(), 'is begin')
    m, n, c = img.shape
    var_array = get_var_list(img, k)
    img_process = np.empty((m, l))
    for i in range(m):
        for j in range(l):
            x = i
            y = j * n / l
            y_int = np.int(y)
            if y_int - 2 < 0:
                y0 = 0
            elif y_int + 4 > n - 1:
                y0 = n - 5
            else:
                y0 = y_int - 2 + np.argmin(var_array[x, y_int - 2:y_int + 1])
            img_process[i, j] = lagrange(img, x, y, k, y0)
    print('process id =', os.getpid(), 'is done')
    return img_process


def func(img, l):
    m, n, c = img.shape
    img_work = np.empty((m, l, c))
    pool = Pool(3)
    result_list = []
    for k in range(3):
        result = pool.apply_async(func=single_process, args=(img, l, k))
        result_list.append(result)
    pool.close()  # 等子进程执行完毕后关闭线程池
    pool.join()
    for k in range(3):
        img_work[::, ::, k] = result_list[k].get()
    return img_work


if __name__ == '__main__':
    path = os.path.abspath('.')
    img = pyplot.imread(path + '/lena512color.tiff')
    row, column = 768, 768
    time_start = time.time()  # 计时开始
    img_temp = np.transpose(func(img, column), axes=(1, 0, 2))  # 做横向插值
    img_done = np.transpose(func(img_temp, row), axes=(1, 0, 2))  # 做纵向插值
    time_end = time.time()  # 计时结束
    print('cost:', time_end - time_start)
    pyplot.imshow(np.uint8(img_done), interpolation="none")
    pyplot.axis('off')
    pyplot.show()
    pyplot.imsave(path + '/lena512color`.tiff', np.uint8(img_done))
