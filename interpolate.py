class Interpolator:
    def __init__(self, matrix, box_size = 9):
        self.m = matrix
        self.len_x = len(self.m[0])
        self.len_y = len(self.m)
        self.box_size = box_size
        self.mid = self.box_size//2 + 1
    
    def get_adj(self, x, y, dx, dy):
        over_x = -(x + dx - self.len_x + 2) if x + dx >= self.len_x else x
        return (self.m[y][over_x])
    
    def get_avg(self, x, y):
        n = self.box_size
        s = 0
        for dy in range(n):
            for dx in range(n):
                s += self.get_adj(x, y, dx - self.mid, dy - self.mid)
        return s/(n**2)

    def interpolate(self):
        m = self.m
        mid = self.mid
        s = self.box_size
        output = []
        for y in range(len(m)):
            if (y % s == mid):
                output.append([])
            for x in range(len(m[0])):
                if (y % s == mid and x % s == mid):
                    output[y//s].append(self.get_avg(x, y))
        return output
    
    def reduce_1d(self, l):
        s = self.box_size
        mid = self.mid
        output = []
        for i in range(len(l)):
            if (i % s == mid):
                output.append(l[i])
        return output