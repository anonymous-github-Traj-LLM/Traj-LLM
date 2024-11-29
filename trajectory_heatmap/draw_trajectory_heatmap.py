import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class TrajectoryDensity:
    def __init__(self) -> None:
        self.data = None
        pass

    def load_data(self, path):
        if path.endswith('.npy'):
            self.data = np.load(path, allow_pickle=True)
        elif path.endswith('.pkl'):
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise ValueError('Unsupported file format')
        
    def filter_density(self, hmap, threshold=1000):
        hmap = np.array(hmap)
        hmap[hmap<threshold] = 0
        return hmap

    def draw_heat_map(self, minx, maxx, miny, maxy, title=None, save_path=None):
        plt.figure(figsize=(10, 8), dpi=100)
        default_cmap = plt.get_cmap('gnuplot2').reversed()
        new_cmap = LinearSegmentedColormap.from_list(
            'custom', 
            default_cmap(np.linspace(0, 0.6, 100))
        )
        hmap = [[0]*(maxx-minx) for _ in range(maxy-miny)]
        data = self.data.reshape((-1,2))
        data = np.floor(data)
        data = data.astype(np.int)
        unique_coords, counts = np.unique(data, axis=0, return_counts=True)
        for i,(x,y) in enumerate(unique_coords):
            if 0<=y-miny<=len(hmap)-1 and 0<=x-minx<=len(hmap[0])-1:
                hmap[y-miny][x-minx] += counts[i]
        filter_threshold = len(data) * 0.001
        hmap = self.filter_density(hmap, threshold=filter_threshold)
        scale_hmap = np.log10(np.array(hmap))
        plt.imshow(scale_hmap, cmap=new_cmap, extent=[minx,maxx,miny,maxy])
        plt.colorbar()
        plt.title(title)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()