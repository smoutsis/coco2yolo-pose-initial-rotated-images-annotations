# coco2yolo-pose-initial-rotated-images-annotations

In this project the txts which are needed for the training of yolov8pose are created from the json files of COCO dataset [1].
Moreover, apart of the initial annotations the rotated coordinates are also created for the rotated images (90, 270 and 180 degrees) of COCO dataset.
Please note that the initial, and the rotated, images are saved again, so take into account your memory.

Firstly, you can download the train and val images from the following urls [ http://images.cocodataset.org/zips/train2017.zip ] and [ http://images.cocodataset.org/zips/test2017.zip ], respectively.
Additionally, the annotations are available to be downloaded from this link: [ http://images.cocodataset.org/annotations/annotations_trainval2017.zip ]
The above urls can be found to the official coco dataset site [ https://cocodataset.org/#home ], at the Dataset section. Note, that the project has been designed for the pose data of 2017.

After you unzip the images train2017.zip, test2017.zip and annotations_trainval2017.zip place them to the same folder. That is the root path where it should be placed on the 148 line of coco2yolo.py script.
Furtermore, create a destination path that will contain two folders, one entitled "images" and the other "labes". Additionally, each folder should contain two more, empty, folders with the names "train" and "val".
The destination path should be given to the 149 line of the coco2yolo.py script.

Before you run the coco2yolo.py script please install libraries from the requirements.txt with the following command:
pip install -r requirements.txt

The coco2yolo.py script is the one that creates the txts (saves them to the "labels" folder you created in the root path) and also saves the initial and rotated images to the destination path. 
If you do not want the rotated annotations and images make the "rotated_flag" from True -> False
Finally, please note that the script runs seperately for the train and validation set. You can choose on which set the script will run by choosing to comment and uncomment the lines 145 and 146. 
To run it for the validation set comment the 145 line and uncomment 146 line. To run it for the training set uncomment the 145 line and comment the 146 line. The script can run for one subset each time. 



[1] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll ́ar, and C. L. Zitnick, “Microsoft COCO: Common Objects in Context,” in proc. Eur. Conf. Computer Vision, pp. 740–755, 2014.
