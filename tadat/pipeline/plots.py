BLUE_M = "#85C1E9"
BLUE_L = "#3498DB"
BLUE_S = "#A9CCE3"
BLUE_STEEL = "steelblue"
GREEN_L = "#2D7C6D"
GREEN_L = "#308776"
GREEN_M = "#45b39d"
GREEN_S = "#48c9b0"
GRAY_L = "#5D6D7E"
GRAY_M = "#808B96"
GRAY_S = "#D5D8DC"
YELLOW = "#F39C12"


GREENS = [GREEN_L, GREEN_M, GREEN_S]
GRAYS = [GRAY_L, GRAY_M, GRAY_S]
BLUES = [BLUE_L, BLUE_M, BLUE_S, BLUE_STEEL ] 
ALL_COLORS = BLUES + GREENS + GRAYS + [YELLOW]
PALETTE_1 =  [YELLOW, GRAY_L, GRAY_M, GRAY_S, BLUE_S, BLUE_M, BLUE_M]
MIN_BAR_DIFF = 0.015

def plot_df(df, ax, x, ys, cols=None, 
            min_y=0, max_y=1, 
            ylabel=None, xlabel=None,
           rot=0, width=0.8, leg=True, 
           annotation_size=None, min_y_diff=0.015,x_pad=4, err=None):    
    
    if err:
        df.plot(ax=ax,x=x,y=ys,kind="bar",color=cols, ylim=[min_y,max_y],label=ys,legend=True,
                rot=rot,colormap=None,width=width, yerr=err)
    else:
        df.plot(ax=ax,x=x,y=ys,kind="bar",color=cols, ylim=[min_y,max_y],label=ys,legend=True,
                rot=rot,colormap=None,width=width)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is None:
        ax.set_xlabel("")    
    else:
        ax.set_xlabel(xlabel)    
    
    if annotation_size is not None:
        annotate_bars(ax, annotation_size, min_y_diff, x_pad)
        
    if leg:
        ax.legend(loc='upper right', bbox_to_anchor=[1.02, 1.05], fancybox=True, shadow=True, ncol=2)
    else:
        ax.legend().set_visible(False)
    return ax

def annotate_bars(ax, annotation_size, min_y_diff, x_pad=4):
    #dictionary with coordinates and values
    patch_dic = {p.get_x():p.get_height() for p in ax.patches}
    #bar width
    width = ax.patches[0].get_width()
    #annotate bars but make sure that close by values do not overlap
    last_val = 0   
    #get an ordered list of bar locations and values
    sorted_patches = [(k,patch_dic[k]) for k in sorted(patch_dic.keys())]    
    for x,y in sorted_patches:    
        dif = y-last_val
        last_val=y
        #skip annotations of similar closeby values
        if min_y_diff > 0 and abs(dif) < min_y_diff: 
            continue    
        y_str = str(round(float(y),3)).replace("0.",".")
        if y > 0:
            ax.annotate(y_str, (x+(width/x_pad), y + (0.01) ),fontsize=annotation_size)
        else:
            ax.annotate(y_str, (x+(width/x_pad), y - (0.02) ),fontsize=annotation_size)

def plot_decay(df, ax, x, ys, cols=None, 
               min_y=0.3, max_y=1,
               ylabel=None,xlabel=None,
               rot=0):    
    df.plot(ax=ax,x=x,y=ys,kind="line",color=cols,
            ylim=[min_y,max_y],label=ys,legend=True,rot=rot,colormap=None, marker='o',markersize=7)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is None:
        ax.set_xlabel("")    
    else:
        ax.set_xlabel(xlabel)    
    for l in ax.lines:
        l.set_linewidth(4)
    #hide first and last ytick labels
    new_ticks = [x for x in ax.get_yticks() if x>=min_y and x<=max_y][1:-1]
    ax.set_yticks(new_ticks)
    #setup legend
    ax.legend(loc='upper right', bbox_to_anchor=[1.02, 1.05], fancybox=True, shadow=True)
    return ax
    