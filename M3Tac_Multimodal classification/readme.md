#Multimodal classification network

In a home or laboratory scenario, we often need to manipulate containers with different textures, shapes, and temperatures, and sometimes the container's excessively high temperature or cold temperature can easily damage human skin.  The use of MTac not only allows us to obtain information about the temperature of the object at the time of contact but also allows us to realize the classification of containers during the grasping process. To realize this function we designed a classification network. we classified and placed the data of each object in the following format:


├── MTac_Multimodal classification algorithm

│   ├── test                
│   │	├── texture      
│   │   ├── temp
│   ├── train                
│   │	├── texture      
│   │   ├── temp
│   │

│   ├── GripperVision.py

│   ├──val_acc-train_acc.pth



