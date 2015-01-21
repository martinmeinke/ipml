'''
Created on Jan 15, 2015

@author: patrik
'''
'''
Created on Jan 13, 2015

@author: patrik
'''
from scipy.weave import inline
import numpy as np
from cmath import sqrt

def cpp_pic_dist(pic, texel):
    h2,w2,d = pic.shape
    delta2, noneed, noneed2 = texel.shape
    maxh = h2-delta2+1
    maxw = w2-delta2+1
    
    dist = inline(cpp_pic_dist_corr,['pic','texel','maxh','maxw','w2'])
    
    dist = dist / (255*sqrt(3))
    dist = np.real(dist)
    return dist

def cpp_tex_dist(t1,t2):
    dist = inline(cpp_texel_dist_corr,['t1','t2'])
    return dist


cpp_texel_dist_corr = """

float val = 0.;
float dist = 0.;
int i = 0;
int j = 0;
int x = 0;
int y = 0;

for(i=0;i<25;i++){
    x = 3*i;
    val = sqrt((t1[x]-t2[x])*(t1[x]-t2[x])+(t1[x+1]-t2[x+1])*(t1[x+1]-t2[x+1])+(t1[x+2]-t2[x+2])*(t1[x+2]-t2[x+2]));
    if(val > dist){
        dist = val;
        }
    }
    
return_val = dist;
"""

cpp_pic_dist_corr = """

float act_max = 441.67296;
int i = 0;
int j = 0;
int x = 0;

for(i=0;i<maxh;i++){
    for(j=0;j<maxw;j++){
        x = i*3*w2+j*3;
        float val = 0.;
        float dist = 0.;
        int a = 0;
        int b = 0;
        int r = 0;
        int t = 0;
        
        for(a=0;a<5;a++){
            for(b=0;b<5;b++){
                r = x+a*3*w2+b*3;
                t = 3*(a*5+b);
                val = sqrt((pic[r]-texel[t])*(pic[r]-texel[t])+(pic[r+1]-texel[t+1])*(pic[r+1]-texel[t+1])+(pic[r+2]-texel[t+2])*(pic[r+2]-texel[t+2]));
                if(val > dist){
                    dist = val;
                    }
                }
            }
            
        if(dist < act_max){
            act_max = dist;
        }
            
        }
    }

return_val = act_max;
"""