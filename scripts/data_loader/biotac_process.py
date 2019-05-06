import numpy as np
from biotac_pos import *

class BtFns(object):
    def __init__(self):
        self.e_pos=get_elect_pos()
        self.e_norm=get_normal_vec()
    def get_force_normal(self,e_data):
        #thresh_indices = e_data < 0.0
        #e_data[thresh_indices] = 0
        electrode_data=e_data#**2
        normal=np.matrix(electrode_data)*self.e_norm
        normal_vec=-1.0*np.ravel(normal)/(np.linalg.norm(normal)+1e-8)
        return normal_vec

    def get_contact_pt(self,e_data,meters=True):
        electrode_data=e_data**2 # squared as per paper
        normalized_data=electrode_data/(np.sum(electrode_data)+1e-8)
        contact_pt=np.matrix(normalized_data)*self.e_pos #[1x19]* [19*3]= [1x3] centroid
        contact_pt=np.ravel(contact_pt)
        v_norm=0.0
        r=5.5/1000.0
        if(contact_pt[0]>0.0):
            for j in range(3):
                v_norm+=contact_pt[j]**2
            contact_pt=contact_pt*(r/np.sqrt(v_norm))
        else:
            j=1
            while(j<3):
                v_norm+=contact_pt[j]**2
                j+=1
            j=1
            while(j<3):
                contact_pt[j]=contact_pt[j]*(r/np.sqrt(v_norm))
                j+=1
        return np.ravel(contact_pt)
    def get_surface_normal(self,cpt):
        # check if contact point is in the cylinder or sphere:
        surface_normal=np.zeros(3)
        if(cpt[0]>0.0):
            # cpt is in the sphere:
            surface_normal=cpt
        else:
            # cpt is in the cylinder    
            surface_normal[0]=0
            surface_normal[1]=cpt[1]
            surface_normal[2]=cpt[2]
            
        surface_normal=surface_normal/(np.linalg.norm(surface_normal)+1e-8)
        return surface_normal
