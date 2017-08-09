import numpy as np
import csv
from collections import Counter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = 'cs3_1N_probe_img.csv'
#filename = 'cs3_1N_gallery_S1.csv'
#filename = 'cs3_1N_gallery_S2.csv'

def func1(x, a, c, d):
    return a*np.exp(-c*x)+d

def func2(x, a, c ,d ):
    return  a*  c * np.exp(-x*c) +d

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
    popt1,pcov1 =  curve_fit(func1, x, y, p0=(1, 1e-6, 1))
    popt2,pcov2 =  curve_fit(func2, x, y, p0=(1, 1e-6, 1))
    #print popt, pcov
    print 'one standard deviation errors: '
    print np.sqrt(np.diag(pcov1)), np.sqrt(np.diag(pcov2))
    
    #print x,y
    indices = np.arange(max(x))
    width = 1
    plt.bar(x, y, width )
    x1 = np.array(indices[1:]).astype(np.int32)
    y1 = np.array(func1(x1,*popt1)).astype(np.int32)
    y2 = np.array(func2(x1,*popt2)).astype(np.int32)
    print x1,y1,y2
    plt.plot(x1, y1, 'g-')
    plt.xticks(indices+1+width*0.5, indices+1)
    plt.xlabel('template size distribution')
    plt.ylabel('number of templates that has x photos')
    plt.title(filename)
    plt.show()
    

filename = 'cs3_1N_gallery_S1.csv'
gallery_dict = dict()
with open("/nfs/isicvlnas01/projects/glaive/data/CS3-tar-files/CS3/protocol/"+filename) as csvfile:
    gallery_reader = csv.reader(csvfile, delimiter = ',')
    next(gallery_reader, None) # skip the header
    for row in gallery_reader:
        template_id = row[0]
        if template_id in gallery_dict:
            gallery_dict[template_id]+=1
        else:
            gallery_dict[template_id] =1

filename = 'cs3_1N_gallery_S2.csv'
with open("/nfs/isicvlnas01/projects/glaive/data/CS3-tar-files/CS3/protocol/"+filename) as csvfile:
    gallery_reader = csv.reader(csvfile, delimiter = ',')
    next(gallery_reader, None) # skip the header
    for row in gallery_reader:
        template_id = row[0]
        if template_id in gallery_dict:
            gallery_dict[template_id]+=1
        else:
            gallery_dict[template_id] =1
        
    cnt = Counter(gallery_dict.values())
    print 'gallery file analyzed. {num_photos_in_template: num_templates}'
    print cnt

    
    x,y = zip(*cnt.items())
    x = x[1:]
    y = y[1:]
    popt1,pcov1 =  curve_fit(func1, x, y, p0=(1, 1e-6, 1))
    popt2,pcov2 =  curve_fit(func2, x, y, p0=( 1, 1e-6, 1))

    #print popt, pcov
    print 'one standard deviation errors: '
    print np.sqrt(np.diag(pcov1)),np.sqrt(np.diag(pcov2))

    #print x,y
    
    indices = np.arange(max(x))
    width = 1
    plt.bar(x, y, width )
    x1 = np.array(indices[1:]).astype(np.int32)
    y1 = np.array(func1(x1,*popt1)).astype(np.int32)
    y2 = np.array(func2(x1,*popt2)).astype(np.int32)
    print x1,y1,y2
    plt.plot(x1, y1, 'g-')
    plt.xticks(indices+1+width*0.5, indices+1)
    plt.xlabel('template size distribution')
    plt.ylabel('number of templates that has x photos')
    plt.title('cs3_1N_gallery')
    plt.show()
    

    

#from 2 to 20
array = [ 371, 257, 178, 123,  86 , 60,  42,  30,  21,  16,  12,   9,   7,   6,   5,   4,   4, 3, 3]
total = sum(array)
print 1.0*array/total
# output:
# array([ 0.29991916,  0.20776071,  0.14389652,  0.09943411,  0.06952304, 0.04850445,  0.03395311,  0.02425222,  0.01697656,  0.01293452, 0.00970089,  0.00727567,  0.00565885,  0.00485044,  0.00404204, 0.00323363,  0.00323363,  0.00242522,  0.00242522])


# from 3 to 20
array = [257, 178, 123,  86 , 60,  42,  30,  21,  16,  12,   9,   7,   6,   5,   4,   4, 3, 3]
total = sum(array)
print 1.0*array/total
# output: 
#[ 0.29676674,  0.20554273,  0.14203233,  0.09930716,  0.06928406,  0.04849885,  0.03464203,  0.02424942,  0.01847575,  0.01385681,  0.01039261,  0.00808314,  0.00692841,  0.00577367,  0.00461894,  0.00461894,  0.0034642,   0.0034642 ]

