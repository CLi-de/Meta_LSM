import math
from skimage import io, color
import numpy as np
from tqdm import trange
from osgeo import gdal
import pandas as pd
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, dem, aspect, curvature, slope):
        self.update(h, w, dem, aspect, curvature, slope)
        self.pixels = []
        self.no = self.cluster_index
        self.cluster_index += 1  # 计数

    def update(self, h, w, dem, aspect, curvature, slope):
        self.h = h
        self.w = w
        self.dem = dem
        self.aspect = aspect
        self.curvature = curvature
        self.slope = slope

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.dem, self.aspect, self.curvature,
                                        self.slope)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        return Cluster(h, w,
                       self.data[0][h][w],
                       self.data[1][h][w],
                       self.data[2][h][w],
                       self.data[3][h][w], )

    def __init__(self, filename, K, M):  # K:number of superpixels; M:衡量像素距离占距离测量的比重
        self.file = filename
        self.K = K
        self.M = M
        # self.data = self.open_image(filename) # shape:(, , 3)
        self.data = self.readTif(filename)  # shape：(6, , )
        self.im_geotrans = gdal.Open(filename).GetGeoTransform()
        self.image_height = self.data.shape[1]
        self.image_width = self.data.shape[2]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)  # np.inf正无穷

    def readTif(self, fileName):
        dataset = gdal.Open(fileName)
        if dataset == None:
            print(fileName + "文件无法打开")
            return
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_bands = dataset.RasterCount  # 波段数
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
        im_proj = dataset.GetProjection()  # 获取投影信息
        # col = int((coor[i][0] - im_geotrans[0]) / im_geotrans[1])
        # row = int((coor[i][1] - im_geotrans[3]) / im_geotrans[5])
        # im_nirBand = im_data[3,0:im_height,0:im_width]#获取近红外波段
        return im_data

    def init_clusters(self, data):
        h = int(self.S / 2)  # 第一个中心点位（cluster）
        w = int(self.S / 2)
        while h < self.image_height:
            while w < self.image_width:
                if data[0][h][w] != -9999:  # -9999为Nodata
                    self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = int(self.S / 2)
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[0][h + 1][w + 1] - self.data[0][h][w] + \
                   self.data[1][h + 1][w + 1] - self.data[1][h][w] + \
                   self.data[2][h + 1][w + 1] - self.data[2][h][w] + \
                   self.data[3][h + 1][w + 1] - self.data[3][h][w]

        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)  # 计算每个中心的gradient
            for dh in range(-5, 6):
                for dw in range(-5, 6):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    if self.data[0][_h][_w] and self.data[1][_h][_w] and self.data[2][_h][_w] \
                            and self.data[3][_h][_w] != -9999:
                        new_gradient = self.get_gradient(_h, _w)
                        if new_gradient < cluster_gradient:  # 寻找 4 x 4 邻域内梯度最小的像素点(更聚集)，并且移动中心
                            cluster.update(_h, _w, self.data[0][_h][_w], self.data[1][_h][_w], self.data[2][_h][_w],
                                           self.data[3][_h][_w])
                            cluster_gradient = new_gradient

    def assignment(self):
        W = [0.3, 0.1, 0.2, 0.4]  # 权重：各因素影响
        for cluster in self.clusters:
            for h in range(cluster.h - self.S, cluster.h + self.S):
                if h < 0 or h >= self.image_height: continue  # continue进入下一个循环
                for w in range(cluster.w - self.S, cluster.w + self.S):
                    if w < 0 or w >= self.image_width: continue
                    if self.data[0][h][w] != -9999 and self.data[1][h][w] != -9999 \
                            and self.data[2][h][w] != -9999 and self.data[3][h][w] != -9999:
                        Dc = math.sqrt(
                            math.pow(self.data[0][h][w] - cluster.dem, 2) * W[0] +
                            math.pow(self.data[1][h][w] - cluster.aspect, 2) * W[1] +
                            math.pow(self.data[2][h][w] - cluster.curvature, 2) * W[2] +
                            math.pow(self.data[3][h][w] - cluster.slope, 2) * W[3]
                        )  # dbs
                        Ds = math.sqrt(
                            math.pow(h - cluster.h, 2) +
                            math.pow(w - cluster.w, 2))  # dxy
                        D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))  # Ds
                        if D < self.dis[h][w]:
                            if (h, w) not in self.label:  # dict中tuple也可以作为key
                                self.label[(h, w)] = cluster
                                cluster.pixels.append((h, w))
                            else:
                                self.label[(h, w)].pixels.remove((h, w))
                                self.label[(h, w)] = cluster
                                cluster.pixels.append((h, w))
                            self.dis[h][w] = D
                        # else:  # 由于分辨率不统一，所以边缘会有各波段不契合的情况，这里统一
                        #     self.data[0][h][w], self.data[1][h][w], self.data[2][h][w],\
                        #     self.data[3][h][w] = [-9999, -9999, -9999, -9999]

    def update_cluster(self):  # 计算各SLIC聚类的中心
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                cluster.update(_h, _w, self.data[0][_h][_w], self.data[1][_h][_w], self.data[2][_h][_w],
                               self.data[3][_h][_w])  # 计算聚类中心

    def writeTiff(self, im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        im_bands, im_height, im_width = im_data.shape
        path = 'seg_output\\' + path
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
        if (dataset != None):
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).SetNoDataValue(-9999)
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset

    def savetif(self, tiffile, savename, image_arr):
        gdal.AllRegister()
        dataset = gdal.Open(tiffile)
        im_bands = dataset.RasterCount  # 波段数
        im_width = dataset.RasterXSize  # 列数
        im_height = dataset.RasterYSize  # 行数
        im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
        im_proj = dataset.GetProjection()  # 获取投影信息

        self.writeTiff(image_arr, im_width, im_height, im_bands, im_geotrans, im_proj, savename)

    def save_current_image(self, tiffile, savename):
        image_arr = np.copy(self.data)
        for i in range(len(image_arr)):  # 初始化
            for h in range(self.image_height):
                for w in range(self.image_width):
                    image_arr[i][h][w] = -9999

        c = 0;
        interval = int(256 / len(self.clusters))
        for cluster in self.clusters:  # 可视化各聚类点
            c += 1
            r = g = b = d = interval * c - 1
            for p in cluster.pixels:  # LAB三通道赋值
                image_arr[0][p[0]][p[1]] = r
                image_arr[1][p[0]][p[1]] = g
                image_arr[2][p[0]][p[1]] = b
            image_arr[0][cluster.h][cluster.w] = 0  # 让cluster中心为0
            image_arr[1][cluster.h][cluster.w] = 0
            image_arr[2][cluster.h][cluster.w] = 0

        self.savetif(tiffile, savename, image_arr)

    def iterate_times(self, loop=5):
        self.init_clusters(self.data)  # 存储所有中心点, clusters = []
        self.move_clusters()
        for i in trange(loop):
            self.assignment()
            self.update_cluster()
            savename = FLAGS.str_region + '_Elegent_Girl_M{m}_K{k}_loop{loop}.tif'.format(loop=i, m=self.M,
                                                                                          k=self.K)  # 生成可视tif
            self.save_current_image(self.file, savename)

    # def show_data(self):
    #     print(self.data[0][0][0])


class TaskSampling(object):
    def __init__(self, clusters):
        self.clusters = clusters
        self.tasks = self.init_tasks(len(clusters))

    def init_tasks(self, num_clusters):
        L = []
        for i in range(num_clusters):
            L.append([])
        return L

    def readpts(self, filepath):
        data = pd.read_excel(filepath, index_col=0, dtype=np.float32)
        data.to_csv('tmp/' + FLAGS.str_region + 'data.csv', encoding='utf-8')
        tmp = np.loadtxt('tmp/' + FLAGS.str_region + 'data.csv', dtype=np.str, delimiter=",", encoding='UTF-8')
        features = tmp[1:, :-3].astype(np.float32)
        features = features / features.max(axis=0)  # 特征归一化
        xy = tmp[1:, -3: -1].astype(np.float32)
        label = tmp[1:, -1].astype(np.float32)
        return features, xy, label

    def sampling(self, im_geotrans):
        features, xy, label = self.readpts(FLAGS.landslide_pts)
        # 计算（row, col）
        pts = []
        for i in range(xy.shape[0]):
            height = int((xy[i][1] - im_geotrans[3]) / im_geotrans[5])
            width = int((xy[i][0] - im_geotrans[0]) / im_geotrans[1])
            pts.append((height, width))

        #  tasks[i].append(height, weight)
        pt_index = 0
        for pt in pts:
            k = 0  # count cluster
            for cluster in self.clusters:
                if (pt[0], pt[1]) in cluster.pixels:
                    self.tasks[k].append([features[pt_index], label[pt_index]])
                    # self.tasks[k].append([xy[pt_index][0], xy[pt_index][1],
                    #                       features[pt_index], label[pt_index]])
                    break
                else:
                    k += 1
            pt_index += 1
        return self.tasks

# if __name__ == '__main__':
#     for k in [192]:
#         p = SLICProcessor("C:\\Users\hj\Desktop\\thematic_map\\CompositeBands2.tif", k, 250)
#         # p.show_data()
#         p.iterate_5times()
#
#         t = TaskSampling(p.clusters)
#         t.sampling(p.im_geotrans)
