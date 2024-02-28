import math

class ToolBox:
    def __init__(self):
        pass

    def remove_list_indexs(self, l, to_remove):
        for i in sorted(to_remove, reverse=True):
            del l[i]
        return l
    
    def nearest_search(self, arr, x):
        low = 0
        high = len(arr) - 1
        mid = 0
        while low <= high:
            mid = (high + low) // 2
            #if there is approximate solution...
            if (high - low <= 2):
                #check nearest 7 indexes for closest match
                poss = [mid + x - 4 for x in range(7)]
                #remove out of bounds indexes
                out_of_bounds = []
                for i in range(len(poss)):
                    if poss[i] < 0 or poss[i] >= len(arr):
                        out_of_bounds.append(i)
                poss = self.remove_list_indexs(poss, out_of_bounds)
                #find closest
                m = mid
                for i in poss:
                    if (abs(arr[i] - x) < abs(arr[m] - x)):
                        m = i
                return m
            # If x is greater, ignore left half
            if arr[mid] < x:
                low = mid + 1
            # If x is smaller, ignore right half
            elif arr[mid] > x:
                high = mid - 1
            # means x is present at mid
            else:
                return mid
        # If we reach here, then the element was not present
        return -1

    def datestr2int(self, s):
        l = s.split("/")
        return (int(l[0]) - 1) * 12 + int(l[1])
    
    def matplot_tooltips(self, ax, fig, sc, annotations):
        annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}".format(" ".join([annotations[n] for n in ind["ind"]]))
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
    
    def in_bounds(self, index, arr):
        return 0 <= index < len(arr)
    
    def get_avgs(self, t_l, x_l, year, windows):
        output = {}
        focus_index = self.nearest_search(t_l, year)
        volacno_threshold = 99999999
        if (focus_index == -1):
            return [output, None]
        focus_year = t_l[focus_index]
        def above_y_min(i, max_yr, w_size):
            if max_yr != -99999999:
                #print(i, max_yr - w_size, max_yr)
                return t_l[i] >= max_yr - w_size
            else:
                return focus_year - w_size // 2 <= t_l[i] <= focus_year + w_size // 2
        for w_size in windows:
            n = max_yr = 1
            my_sum = float(x_l[focus_index])
            if len(t_l) < w_size:
                continue
            def search_bounds(i, t_l, x_l, w_size, max_yr, n, my_sum, volacno_threshold, direc):
                max_yr = -99999999
                while self.in_bounds(i, t_l) and above_y_min(i, max_yr, w_size):
                    n += 1
                    x = float(x_l[i])
                    if math.isnan(x):
                        x = 0
                        n -= 1
                    if x > volacno_threshold:
                        break
                    my_sum += x
                    max_yr = t_l[i] if t_l[i] > max_yr else max_yr
                    i += direc
                return [n, my_sum, max_yr]
            [n, my_sum, max_yr] = search_bounds(focus_index + 1, t_l, x_l, w_size, max_yr, n, my_sum, volacno_threshold, 1)
            [n, my_sum, max_yr] = search_bounds(focus_index - 1, t_l, x_l, w_size, max_yr, n, my_sum, volacno_threshold, -1)
            output[w_size] = my_sum / n
        return [output, t_l[focus_index]]