if __name__ == "__main__":
    import numpy as np
    depth_list = [3, 4, 23, 3]
    fpn_list = []
    for i in range(len(depth_list)):
        fpn_list.append(np.sum(depth_list[:i+1]))
    print(fpn_list)