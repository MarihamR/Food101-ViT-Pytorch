from torchvision import datasets, transforms, models
import torch
import os

def dataset(data_dir='./food101',batch_size=128,Transformation=False,resize=224,smallset=False):

	train_transforms = transforms.Compose([transforms.RandomRotation(30),
		                               transforms.Resize((resize,resize)),
		                               transforms.RandomHorizontalFlip(),
		                               transforms.ToTensor(),
		                               transforms.Normalize([0.485, 0.456, 0.406],
		                                                    [0.229, 0.224, 0.225])])


	test_transforms = transforms.Compose([transforms.Resize((resize,resize)),
		                              #transforms.CenterCrop(resize),
		                              transforms.ToTensor(),
		                              transforms.Normalize([0.485, 0.456, 0.406],
		                                                   [0.229, 0.224, 0.225])])
	if not Transformation:
		Train_set=datasets.Food101(data_dir,split="train",transform=transforms.ToTensor(),download=False)
		Test_set=datasets.Food101(data_dir,split="test",transform=transforms.ToTensor(),download=False)
	else:
		Train_set=datasets.Food101(data_dir,split="train",transform=train_transforms,download=False)
		Test_set=datasets.Food101(data_dir,split="test",transform=test_transforms,download=False)
        
	if not smallset:  
		Trainloader = torch.utils.data.DataLoader(Train_set, batch_size=batch_size, shuffle=True)
		Testloader=torch.utils.data.DataLoader(Test_set, batch_size=batch_size,shuffle=False)
		return Train_set,Test_set,Trainloader,Testloader
	else:
		Test_set, _ = torch.utils.data.random_split(Test_set, [5250, 20000])
		Testloader=torch.utils.data.DataLoader(Test_set, batch_size=batch_size,shuffle=False)
		return Test_set,Testloader
