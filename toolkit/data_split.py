import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

all_bind = pd.read_csv("../data/all_bind.csv")
pos_bind = pd.DataFrame(all_bind[["pubchem_cid", "gene_id"]])  # 20565
drug = np.unique(all_bind[["pubchem_cid"]])  # 5135
target = np.unique(all_bind[["gene_id"]])  # 2894

all_drug_target = pd.DataFrame(np.repeat(drug, len(target)), columns=["pubchem_cid"])  # 14860690
all_drug_target["gene_id"] = np.tile(target, len(drug))
all_neg_bind = pd.concat([all_drug_target, pos_bind]).drop_duplicates(keep=False)  # 14840125

# 赋予标签
pos_bind['label'] = np.repeat(1, len(pos_bind))
all_neg_bind["label"] = np.repeat(0, len(all_neg_bind))


# 划分训练：验证：测试= 7：1：2
def creat_train_val_test(pos_data, neg_data, random_state):
    neg_data = neg_data.sample(n=len(pos_data), replace=False, random_state=random_state)
    bind_train_pos, bind_val_test_pos = train_test_split(pos_data, test_size=0.3, random_state=random_state)
    bind_val_pos, bind_test_pos = train_test_split(bind_val_test_pos, test_size=2/3, random_state=random_state)
    bind_train_neg, bind_val_test_neg = train_test_split(neg_data, test_size=0.3, random_state=random_state)
    bind_val_neg, bind_test_neg = train_test_split(bind_val_test_neg, test_size=2/3, random_state=random_state)

    bind_train = pd.concat([bind_train_pos, bind_train_neg])
    bind_val = pd.concat([bind_val_pos, bind_val_neg])
    bind_test = pd.concat([bind_test_pos, bind_test_neg])

    return bind_train, bind_val, bind_test


bind_train_part1, bind_val_part1, bind_test_part1 = creat_train_val_test(pos_bind, all_neg_bind, random_state=10)
bind_train_part2, bind_val_part2, bind_test_part2 = creat_train_val_test(pos_bind, all_neg_bind, random_state=20)
bind_train_part3, bind_val_part3, bind_test_part3 = creat_train_val_test(pos_bind, all_neg_bind, random_state=30)
bind_train_part4, bind_val_part4, bind_test_part4 = creat_train_val_test(pos_bind, all_neg_bind, random_state=40)
bind_train_part5, bind_val_part5, bind_test_part5 = creat_train_val_test(pos_bind, all_neg_bind, random_state=50)

save_file = [bind_train_part1, bind_val_part1, bind_test_part1, bind_train_part2, bind_val_part2, bind_test_part2,
             bind_train_part3, bind_val_part3, bind_test_part3, bind_train_part4, bind_val_part4, bind_test_part4,
             bind_train_part5, bind_val_part5, bind_test_part5]

save_file_name = ["bind_train_part1", "bind_val_part1", "bind_test_part1", "bind_train_part2", "bind_val_part2",
                  "bind_test_part2", "bind_train_part3", "bind_val_part3", "bind_test_part3", "bind_train_part4",
                  "bind_val_part4", "bind_test_part4", "bind_train_part5", "bind_val_part5", "bind_test_part5"]

for i in range(0, 15):
    save_file[i].to_csv("demo/warm_start" + str(save_file_name[i]) + ".csv", index=False)
