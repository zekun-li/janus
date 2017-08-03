import numpy as np
import csv
from collections import Counter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#filename = 'cs3_1N_probe_img.csv'
filename = 'cs3_1N_gallery_S1.csv'
#filename = 'cs3_1N_gallery_S2.csv'

def func(x, a, c, d):
    return a*np.exp(-c*x)+d

prob_dict = dict() # dictionary to count occurence of templates
with open("/nfs/isicvlnas01/projects/glaive/data/CS3-tar-files/CS3/protocol/"+filename) as csvfile:
    probe_reader = csv.reader(csvfile, delimiter = ',')
    next(probe_reader, None) # skip the header
    for row in probe_reader:
        #print row
        template_id = row[0]
        if template_id in prob_dict:
            prob_dict[template_id] +=1
        else:
            prob_dict[template_id] = 1

    cnt = Counter(prob_dict.values())
    print filename,'analyzed. {num_photos_in_template: num_templates}'
    print cnt

    x,y = zip(*cnt.items())
    x = x[1:]
    y = y[1:]
    #popt,pcov =  curve_fit(func, x, y, p0=(1, 1e-6, 1))
    #print x,y
    indices = np.arange(len(x))[::5]
    width = 1
    plt.bar(x, y, width )
    #x1 = np.array(x).astype(np.int32)
    #y1 = np.array(func(x1,*popt)).astype(np.int32)
    #plt.plot(x1, y1, 'g-')
    plt.xticks(indices+width*0.5, x[::5])
    plt.ylabel('number of templates that has x photos')
    plt.title(filename)
    plt.show()

    
