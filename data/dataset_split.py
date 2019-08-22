all_ids = list(range(33))
train_segmap_ids = [0, 1, 2, 17]
train_class_ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32]
val_ids = [3, 26, 14, 18]
test_ids = [4, 27, 20, 19]

assert sorted(train_segmap_ids + train_class_ids + val_ids + test_ids) == all_ids, 'Data split is not correct!'

if __name__== "__main__":
    print('all_ids {} = {:.2%}'.format(len(all_ids), len(all_ids) / len(all_ids)))
    print('train_segmap_ids {} = {:.2%}'.format(len(train_segmap_ids), len(train_segmap_ids) / len(all_ids)))
    print('train_class_ids {} = {:.2%}'.format(len(train_class_ids), len(train_class_ids) / len(all_ids)))
    print('val_ids {} = {:.2%}'.format(len(val_ids), len(val_ids) / len(all_ids)))
    print('test_ids {} = {:.2%}'.format(len(test_ids), len(test_ids) / len(all_ids)))

