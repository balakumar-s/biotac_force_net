import numpy as np
def get_elect_pos(): #in mm
    elect_position=np.zeros((20,4))
    elect_position[1][1] = 0.993;
    elect_position[1][2] = -4.855;
    elect_position[1][3] = -1.116;
    elect_position[2][1] = -2.700;
    elect_position[2][2] = -3.513;
    elect_position[2][3] = -3.670;
    elect_position[3][1] = -6.200;
    elect_position[3][2] = -3.513;
    elect_position[3][3] = -3.670;
    elect_position[4][1] = -8.000;
    elect_position[4][2] = -4.956;
    elect_position[4][3] = -1.116;
    elect_position[5][1] = -10.500;
    elect_position[5][2] = -3.513;
    elect_position[5][3] = -3.670;
    elect_position[6][1] = -13.400;
    elect_position[6][2] = -4.956;
    elect_position[6][3] = -1.116;
    elect_position[7][1] = 4.763;
    elect_position[7][2] = 0.000;
    elect_position[7][3] = -2.330;
    elect_position[8][1] = 3.031;
    elect_position[8][2] = -1.950;
    elect_position[8][3] = -3.330;
    elect_position[9][1] = 3.031;
    elect_position[9][2] = 1.950;
    elect_position[9][3] = -3.330;
    elect_position[10][1] = 1.299;
    elect_position[10][2] = 0.000;
    elect_position[10][3] = -4.330;
    elect_position[11][1] = 0.993;
    elect_position[11][2] = 4.855;
    elect_position[11][3] = -1.116;
    elect_position[12][1] = -2.700;
    elect_position[12][2] = 3.513;
    elect_position[12][3] = -3.670;
    elect_position[13][1] = -6.200;
    elect_position[13][2] = 3.513;
    elect_position[13][3] = -3.670;
    elect_position[14][1] = -8.000;
    elect_position[14][2] = 4.956;
    elect_position[14][3] = -1.116;
    elect_position[15][1] = -10.500;
    elect_position[15][2] = 3.513;
    elect_position[15][3] = -3.670;
    elect_position[16][1] = -13.400;
    elect_position[16][2] = 4.956;
    elect_position[16][3] = -1.116;
    elect_position[17][1] = -2.800;
    elect_position[17][2] = 0.000;
    elect_position[17][3] = -5.080;
    elect_position[18][1] = -9.800;
    elect_position[18][2] = 0.000;
    elect_position[18][3] = -5.080;
    elect_position[19][1] = -13.600;
    elect_position[19][2] = 0.000;
    elect_position[19][3] = -5.080;
    return elect_position[1:20,1:4]/1000.0 # convert to meters

def get_normal_vec():
    elect_normal=np.zeros((20,4))
    elect_normal[1][1] = 0.196;
    elect_normal[1][2] = -0.956;
    elect_normal[1][3] = -0.220;#0.220;
    elect_normal[2][1] = 0.0;
    elect_normal[2][2] = -0.692;
    elect_normal[2][3] = -0.722;
    elect_normal[3][1] = 0.0;
    elect_normal[3][2] = -0.692;
    elect_normal[3][3] = -0.722;
    elect_normal[4][1] = 0.0;
    elect_normal[4][2] = -0.976;
    elect_normal[4][3] = -0.220;
    elect_normal[5][1] = 0.0;
    elect_normal[5][2] = -0.692;#-0.976;
    elect_normal[5][3] = -0.722;#-0.220;
    elect_normal[6][1] = 0.0;#0.5;
    elect_normal[6][2] = -0.976;#0.0;
    elect_normal[6][3] = -0.220;#-0.866;
    elect_normal[7][1] = 0.5;
    elect_normal[7][2] = 0.0;
    elect_normal[7][3] = -0.866;
    elect_normal[8][1] = 0.5;
    elect_normal[8][2] = 0.0;
    elect_normal[8][3] = -0.866;
    elect_normal[9][1] = 0.5;
    elect_normal[9][2] = 0.0;
    elect_normal[9][3] = -0.866;
    elect_normal[10][1] = 0.5;
    elect_normal[10][2] = 0.0;
    elect_normal[10][3] = -0.866;
    elect_normal[11][1] = 0.196;
    elect_normal[11][2] = 0.956;
    elect_normal[11][3] = -0.220;
    elect_normal[12][1] = 0.0;
    elect_normal[12][2] = 0.692;
    elect_normal[12][3] = -0.722;
    elect_normal[13][1] = 0.0;
    elect_normal[13][2] = 0.692;
    elect_normal[13][3] = -0.722;
    elect_normal[14][1] = 0.0;
    elect_normal[14][2] = 0.976;
    elect_normal[14][3] = -0.220;
    elect_normal[15][1] = 0.0;
    elect_normal[15][2] = 0.692;
    elect_normal[15][3] = -0.722;
    elect_normal[16][1] = 0.0;
    elect_normal[16][2] = 0.976;
    elect_normal[16][3] = -0.220;
    elect_normal[17][1] = 0.0;
    elect_normal[17][2] = 0.0;
    elect_normal[17][3] = -1.000;
    elect_normal[18][1] = 0.0;
    elect_normal[18][2] = 0.0;
    elect_normal[18][3] = -1.000;
    elect_normal[19][1] = 0.0;
    elect_normal[19][2] = 0.0;
    elect_normal[19][3] = -1.000;
    return 1.0*elect_normal[1:20,1:4]

