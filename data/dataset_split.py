"""Train/Validation/Test dataset split

The dataset needs to be split into train/val/test sets to properly validate the hypothesis.
To make sure the split is always consistent, it is hardcoded here.
All other scripts reference this file for the split ids.
"""

all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38]
train_strong_ids = [1, 2, 3, 21]
train_weak_ids = [6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 20, 26, 27, 28, 29, 30, 33, 34, 35, 37, 38]
val_ids = [4, 16, 22, 31]
test_ids = [5, 23, 24, 32]

assert sorted(train_strong_ids + train_weak_ids + val_ids + test_ids) == all_ids, 'Data split is not correct!'

if __name__== "__main__":
    print('all_ids {} = {:.2%}'.format(len(all_ids), len(all_ids) / len(all_ids)))
    print('train_strong_ids {} = {:.2%}'.format(len(train_strong_ids), len(train_strong_ids) / len(all_ids)))
    print('train_weak_ids {} = {:.2%}'.format(len(train_weak_ids), len(train_weak_ids) / len(all_ids)))
    print('val_ids {} = {:.2%}'.format(len(val_ids), len(val_ids) / len(all_ids)))
    print('test_ids {} = {:.2%}'.format(len(test_ids), len(test_ids) / len(all_ids)))

