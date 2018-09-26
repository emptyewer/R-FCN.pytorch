### Kaggle Pneumonia Data Directory

- Create a directory `data` NOTE: Dockerfile should create `data` directory in cloned repository directory, e.g. `R-FCN/data`, this can be used for bind mount
- Download Kaggle Pneumonia Data
- Create directory recursively: `data/PNAdevkit/PNA2018` (PNA stands for Pneumonia), and in the `PNA2018` directory create the following directories:
	- `DCMImagesTrain` - This will contain `<patientId>.dcm`. Unzip `stage_1_train_images.zip` contents in it.
	- `DCMImagesTest` - This will contain `<patientId>.dcm`. Unzip `stage_1_test_images.zip` contents in it.
	- `ImageSets` - This will contain `train.txt`, `trainval.txt` and `test.txt`, each of these text file will contain `<patientId>` in separate lines.

	```python
	data_dir = '/home/<user>/data/PNAdevkit/PNA2018'
	d = os.path.join(data_dir, 'DCMImagesTest')
	pids = [pid.split('.')[0] for pid in os.listdir(d)]
	
	with open(data_dir + '/ImageSets/' + 'test.txt', 'w') as f:
	    for pid in pids:
	        f.write("{}\n".format(pid))
     
    # Do same for DCMImagesTrain to generate 'trainval.txt'
   
    from sklearn.model_selection import train_test_split
    train_val_df = df.groupby('patientId').apply(lambda x: x.sample(1))
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=2018, 
                                        stratify=train_val_df['class'])
    
    train_pids = train_df['patientId'].tolist()
    with open('data/PNAdevkit/PNA2018/ImageSets/train.txt', 'w') as f:
        for pid in train_pids:
            f.write("{}\n".format(pid))
            
    val_pids = val_df['patientId'].tolist()
    with open('data/PNAdevkit/PNA2018/ImageSets/val.txt', 'w') as f:
        for pid in val_pids:
        f.write("{}\n".format(pid))
	```

	- `Annotations` - This will contain a **`train_labels_bboxes.csv`** file. Steps on creating this file are shown below:
		- Unzip `stage_1_detailed_class_info.csv.zip` 
		- Unzip `stage_1_train_labels.csv.zip` 

	```python
	   # Read class info and labels
    dfl = pd.read_csv('stage_1_train_labels.csv')
    dfd = pd.read_csv('stage_1_detailed_class_info.csv')
    
    # Merge dataframes
    df = pd.merge(left=dfl, right=dfd, on='patientId', how='inner')
    df = df.drop_duplicates()
    df['category'] = df['Target'].apply(lambda t: 'opacity' if t == 1 else 'normal')
    df.reset_index(drop=True, inplace=True)
    df.fillna(0, inplace=True)
    
    # Write file
    df.to_csv('train_labels_bboxes.csv', index=False)
	```

- NOTE: I had to change permissions of the files as I was not able to read them, e.g. `chmod -R 775 data`


### Pre-Trained `ResNet` Models

- Create a directory **`pretrained_model`** in `data` directory

- Pre-trained models:
	
	- **ResNet-18:** `https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth`
	- **ResNet-34:** `https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth`
	- **ResNet-50:** `https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth`
	- **ResNet-101:** `https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth`
	- **ResNet-152:** `https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth`
	
- After downloading the models, they need to be **renamed**. Renaming pre-trained models example:
	- `resnet101-5d3b4d8f.pth` renamed to `resnet101_rcnn.pth` for **rfcn** and **couplenet**, for **rcnn** (Faster R-CNN), `TODO`
	- Similarly after downloading and renaming the models the `pretrained_model` will have the following models:
	    - `resnet101_rcnn.pth` - No issues except weird `nan`, which doesn't happen when using large trainval.txt
	    - `resnet152_rcnn.pth` - No issues except weird `nan`
	    - `resnet18_rcnn.pth` - **Dimension error**, so use `pretrained=False` in `trainval_net.py` while training and testing, however the terminal shows `https://github.com/pytorch/pytorch/issues/4871`.
	    - `resnet34_rcnn.pth` - **Dimension error**, so use `pretrained=False` in `trainval_net.py` while training and testing, however the terminal shows `https://github.com/pytorch/pytorch/issues/4871`.
	    - `resnet50_rcnn.pth` - **Problem loading pre-trained weights**, so use `pretrained=False` in `trainval_net.py` while training and testing, however the terminal shows `https://github.com/pytorch/pytorch/issues/4871`.
	    
    - NOTE: Using `pretrained=False` causes `rcnn_cls` loss and `rcnn_box` loss to be `nan`
    
```
Exception ignored in: <bound method DataLoaderIter.__del__ of <torch.utils.data.dataloader.DataLoaderIter object at 0x7f2b9d889160>>
Traceback (most recent call last):
  File "/usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 333, in __del__
  File "/usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 319, in _shutdown_workers
  File "/usr/local/lib/python3.6/multiprocessing/queues.py", line 337, in get
ImportError: sys.meta_path is None, Python is likely shutting down
```
	    
	    
	    