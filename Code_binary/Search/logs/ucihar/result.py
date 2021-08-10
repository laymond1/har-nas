import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# os.chdir(os.path.join(os.getcwd(), 'Code', 'Search'))

losses = np.zeros([99, 35])
ce_losses = np.zeros([99, 35])
reg_losses = np.zeros([99, 35])
flopses = np.zeros([99, 35])

log_path = "./logs/ucihar/warmp20_gr/logs/"

with open(log_path + "gradient.log", "r") as f:
    lines = f.readlines()
    l = 0
    for i, line in enumerate(lines):
        # print(line)
        
        if "Loss" in line:
            loss = line.split('\t')[-4][5:]
            ce_loss = line.split('\t')[-3][7:]
            reg_loss = line.split('\t')[-2][10:]
            flops = line.split('\t')[-1][6:]
            # print(line.split('\t'))
            # print(float(ce_loss))
            losses[l][i%35] = float(loss)
            ce_losses[l][i%35] = float(ce_loss)
            reg_losses[l][i%35] = float(reg_loss)
            flopses[l][i%35] = round(float(flops) * 1e-6, 1)
            if (i+1) % 35 == 0:
                l += 1
            # break

entropy = np.zeros(99)
with open(log_path + "valid_console.txt", "r") as f:
    lines = f.readlines()
    l = 0
    for i, line in enumerate(lines):
        if 'Warmup' not in line: # i >= 31:
            ent = line.split('\t')[-3][8:]
            print(line.split('\t'))
            entropy[l] = float(ent)
            l += 1
            # print(ent)
            # break
print(entropy)

# print(losses)
# print(ce_losses)
# print(reg_losses)
# print(np.mean(losses, axis=1).shape)
# print(np.mean(ce_losses, axis=1).shape)
# print(np.mean(reg_losses, axis=1).shape)

df = pd.DataFrame(
    {
        'Loss': np.mean(losses, axis=1),
        'CE-Loss': np.mean(ce_losses, axis=1),
        'Reg-Loss': np.mean(reg_losses, axis=1),
        'Flops': np.mean(flopses, axis=1),
        'Entropy': entropy
    }
)
df.to_excel(log_path + 'reg_add.xlsx')
print(df)

plt.style.use('default')

index = np.arange(1, 100)

plt.figure(figsize=(12,6))
l_plot = plt.plot(df["Loss"], 'r')
ce_plot = plt.plot(df["CE-Loss"], 'b')
reg_plot = plt.plot(df["Reg-Loss"], 'g')
# ent_plot = plt.plot(df["Entropy"], 'y')

plt.legend(['Loss','CE-Loss','Reg-Loss']) 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.title('Cifar10 & Flops & Add Term')
plt.axis([0, 100, 0, 3.5])
plt.rc('legend', fontsize=15)  # 범례 폰트 크기 
plt.grid(True) 
plt.savefig(log_path + 'fig1.png', dpi=300)
plt.show()

plt.figure(figsize=(12,6))
l_plot = plt.plot(index, df["Loss"], 'r')
plt.xlabel('Epochs', fontsize=15) 
plt.ylabel('Loss', fontsize=15) 
plt.title('Total Loss', fontsize=20)
# plt.axis([0, 120, 0, 1]) 
plt.grid(True) 
plt.show()

plt.figure(figsize=(12,6))
ce_plot = plt.plot(df["CE-Loss"], 'b')
plt.xlabel('Epochs', fontsize=15) 
plt.ylabel('Loss', fontsize=15) 
plt.title('CE-Loss', fontsize=20)
# plt.axis([0, 100, 0, 1]) 
plt.grid(True) 
plt.show()

plt.figure(figsize=(12,6))
plt.plot(index, df["Reg-Loss"], 'g')
plt.xlabel('Epochs', fontsize=15) 
plt.ylabel('Loss', fontsize=15) 
plt.title('Reg-Loss', fontsize=20)
# plt.axis([0, 120, 0, 1]) 
plt.grid(True) 
plt.savefig(log_path + 'fig2.png', dpi=300)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(index, df["Entropy"], 'y')
plt.xlabel('Epochs', fontsize=15) 
plt.ylabel('Entropy', fontsize=15) 
plt.title('Entropy', fontsize=20)
# plt.axis([0, 120, 0, 1]) 
plt.grid(True) 
plt.savefig(log_path + 'fig3.png', dpi=300)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(index, df["Flops"])
plt.xlabel('Epochs') 
plt.ylabel('Flops(M)') 
plt.title('Flops')
# plt.axis([0, 120, 0, 1]) 
plt.grid(True) 
plt.savefig(log_path + 'fig4.png', dpi=300)
plt.show()
