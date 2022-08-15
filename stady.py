import torch
t1=torch.tensor([1,2,3,4,5])
print(t1.shape)
t2=torch.tensor([1,2,2,4,4])
print(t2)
t3=(t1==t2).squeeze()
print(t3)
c=0
print(t3.shape)
t4=t3.reshape(-1)
print(t4.shape)
for j in t4:
    element1=j.item()
    print(element1)
for i in range(len(t4)):
    c+=t4[i].item()
    print(t4[i].item())
print(c)