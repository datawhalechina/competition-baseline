def check_label(s):
    if '*' in s:
        return True
    if len(s) != 10:
        return True
    
    if len(set(s[3:]) & set(string.ascii_uppercase)) > 0:
        return True
    
    if s[0] in string.digits:
        return True
    
    if s[0] in string.ascii_uppercase and s[1] in string.ascii_uppercase and s[2] in string.ascii_uppercase:
        return True
    
    if s[0] in string.ascii_uppercase and s[1] in string.ascii_uppercase:
        return True 
    elif s[0] in string.ascii_uppercase and s[2] in string.ascii_uppercase and s[1] in string.digits:
        return True
    else:
        return False
    

import pandas as pd
import string
submit_df1 = pd.read_csv('./tmp_rcnn_tta10_pb.csv')
submit_df2 = pd.read_csv('../multi-digit-pytorch/tmp_rcnn_tta10_cnn.csv')

submit_df1.loc[submit_df1['name'] == 'OFTUHPVE.jpg', 'label'] = submit_df2[submit_df2['name'] == 'OFTUHPVE.jpg']['label']
submit_df1[~submit_df1['label'].apply(lambda x: check_label(x))]
submit_df1.to_csv('tmp_rcnn_tta10_pb_submit.csv',index=None)